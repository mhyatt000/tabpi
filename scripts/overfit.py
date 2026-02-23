from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import imageio
import libero
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
import tyro
import wandb

from tabpi.envs.libero import EnvFactory, LiberoFactory
from tabpi.utils.data import check_download, split
from tabpi.utils.deco import timeit
from tabpi.utils.wab import Wandb


@dataclass
class Config:
    task_suite: str = "libero_object"
    task_id: int = 0
    steps: int = 400
    training: float = 1

    demo: int = 0

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)


data_dir = Path(libero.__file__).parents[0] / "datasets"


def main(cfg: Config):
    task_name = cfg.env.get_benchmark(cfg.task_suite).get_task(cfg.task_id).name

    demo_path = cfg.env.get_data_path(data_dir)
    print(demo_path)
    with h5py.File(demo_path, "r") as f:
        states = f[f"data/demo_{cfg.demo}/states"][()]
        actions_demo = f[f"data/demo_{cfg.demo}/actions"][()]
        init_state = states[0]

    # Needs to be same num of rows as env.n_envs
    init_states = np.stack([init_state] * 4)
    print(f"Inits shape: {init_states.shape}")
    venv = cfg.env.build()
    venv.reset()
    obs = venv.set_init_state(init_states)

    print(data_dir)
    check_download(data_dir, cfg.task_suite)

    features = states
    actions = actions_demo
    print(features.shape)
    print(actions.shape)

    x_fit, x_test, y_fit, y_test = split(cfg.training, 0.1, features, actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        # n_estimators=4,
        device=device,
        # device='cuda:0',
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

    print("Predicting on training and testing data")
    predict = timeit(model.predict)
    _, yh_train = predict(x_fit)
    prediction_time, yh_test = predict(x_test)

    mse = mean_squared_error(y_test, yh_test)
    r2_train = r2_score(y_fit, yh_train)
    r2_test = r2_score(y_test, yh_test)
    print("Mean Squared Error (MSE):", mse)
    print("R² Train Score:", r2_train)
    print("R² Test Score:", r2_test)

    print("Initializing Wandb")
    run = cfg.wandb.initialize(cfg)

    frames = []
    steps = 0

    vid_path = "ObsVids/"
    dir_path = Path(vid_path)
    dir_path.mkdir(exist_ok=True)

    while steps < actions_demo.shape[0]:
        print(f"Repeating demo_{cfg.demo} actions_demo{steps}")
        action = np.stack([actions_demo[steps]] * 4)
        obs, venv_rewards, done, _info = venv.step(action)
        done = np.array(done)
        print(f"Done: {done}")

        frames.append(obs[0]["galleryview_image"][::-1])
        steps += 1

    imageio.mimsave(f"{vid_path}Demo_{cfg.demo}{task_name}All.mp4", frames, fps=30)
    print(f"{vid_path}Demo_{cfg.demo}{task_name}All.mp4 saved")

    cfg.wandb.log({f"demo/video_demo{cfg.demo}": wandb.Video(f"{vid_path}Demo_0{task_name}All.mp4", format="mp4")})

    frames = []
    venv.reset()
    total_time, total_success = 0, 0
    done_global, steps, max_steps = False, 0, cfg.steps

    while not done_global and steps < max_steps:
        steps += 1
        print(f"Steps={steps}")

        states = init_states if steps == 1 else np.array(venv.get_sim_state())

        print(f"Predicting actions across {states.shape[0]} envs")
        predict_venv_actions = timeit(model.predict)
        avg_iteration_time, actions = predict_venv_actions(states)
        total_time += avg_iteration_time
        print(f"Avg inference time={avg_iteration_time}")

        obs, venv_rewards, done, _info = venv.step(actions)
        done = np.array(done)
        print(f"Done: {done}")
        success = venv.check_success()
        print(f"Check Success: {success}")
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

        cfg.wandb.log({"Avg Gripper Pos": actions[:, -1]})

    avg_time = total_time / steps
    avg_sr = total_success / steps

    venv.close()

    # Save video
    imageio.mimsave(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4", frames, fps=30)
    print(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4 saved")

    cfg.wandb.log(
        {
            "Fit Time": fit_time,
            "Prediction Time": prediction_time,
            "MSE": mse,
            "R^2": r2_test,
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
