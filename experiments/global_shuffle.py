from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio
import libero
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
import tyro

from tabpi.utils.deco import timeit
from tabpi.utils.util import check_download, EnvFactory, extract, LiberoFactory, split
from tabpi.wab import Wandb
import wandb


@dataclass
class Config:
    task_suite: str = "libero_object"
    task_id: int = 0
    steps: int = 400
    training: float = 0.10

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)


data_dir = Path(libero.__file__).parents[0] / "datasets"


def main(cfg: Config):
    print("Initializing Wandb")
    run = cfg.wandb.initialize(cfg)

    venv = cfg.env.build()
    _ = venv.reset()

    print(data_dir)
    check_download(data_dir, cfg.task_suite)

    raw: dict[str, Any] = cfg.env.load_data(data_dir)
    features, actions = extract(raw)
    print(features.shape)
    print(actions.shape)

    print("Globally Shuffled")
    rng = np.random.default_rng(seed=42)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    features = features[indices]
    actions = actions[indices]

    x_fit, x_test, y_fit, y_test = split(cfg.training, 0.1, features, actions)

    act_dim = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        # n_estimators=act_dim,
        device=device,
        # device='cuda:0',
        fit_mode="fit_with_cache",
        # n_preprocessing_jobs=act_dim*2,
        # memory_saving_mode=False,
        inference_precision="autocast",
    )

    """
    # If multiple GPUs are detected, wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        model.executor_.model = torch.nn.DataParallel(model.executor_.model)
        # model.executor_.model.to(device)
        # from https://github.com/PriorLabs/TabPFN/issues/215

    model.executor_.model.to(device)
    """

    print(f"Fitting on {cfg.training * 100}%")
    fit = timeit(model.fit)
    fit_time, _ = fit(x_fit, y_fit)

    print("Predicting on last 10%")
    predict = timeit(model.predict)
    prediction_time, yh = predict(x_test)
    print(yh.shape)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    quit()

    frames = []
    total_time, total_success = 0, 0
    done_global, steps, max_steps = False, 0, cfg.steps

    vid_path = "ObsVids/"
    dir_path = Path(vid_path)
    dir_path.mkdir(exist_ok=True)

    while not done_global and steps < max_steps:
        steps += 1
        print(f"Steps={steps}")

        states = np.array(venv.get_sim_state())

        print(f"Predicting actions across {states.shape[0]} envs")
        predict_venv_actions = timeit(model.predict)
        avg_iteration_time, actions = predict_venv_actions(states)
        total_time += avg_iteration_time
        print(f"Avg inference time={avg_iteration_time}")

        print(actions.shape)
        obs, venv_rewards, done, _info = venv.step(actions)
        done = np.array(done)
        venv_rewards = np.array(venv_rewards)

        frames.append(obs[0]["galleryview_image"][::-1])

        successes = venv_rewards.sum(axis=-1)
        avg_sr = successes.mean()
        total_success += avg_sr
        print(f"Avg success rate: {avg_sr}")

        # step() sets done=self._check_success()
        done_indices = np.where(done)[0]
        if len(done_indices) > 0:
            print(f"Task completed successfully in venv {done_indices}")
            done_global = True

    avg_time = total_time / steps
    avg_sr = total_success / steps

    venv.close()

    # Save video
    task_name = cfg.env.get_benchmark(cfg.task_suite).get_task(cfg.task_id).name
    imageio.mimsave(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4", frames, fps=30)
    print(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4 saved")

    cfg.wandb.log(
        {
            "Fit Time": fit_time,
            "Prediction Time": prediction_time,
            "MSE": mse,
            "R^2": r2,
            "Average Inference Time": avg_time,
            "Average Reward": avg_sr,
            "Steps until done": steps,
            f"sim/video_{cfg.training * 100}%": wandb.Video(
                f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4", format="mp4"
            ),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
