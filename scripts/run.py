from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import libero
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
from tabpi.utils.util import check_download, EnvFactory, extract, LiberoFactory, split
from tabpi.wab import Wandb
import torch
import tyro
import wandb

from tabpi.utils.timer import Timer


@dataclass
class Config:
    task_suite: str = "libero_object"
    task_id: int = 0
    steps: int = 400
    training: float = 0.10

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)


data_dir = Path(libero.__file__).parents[0] / "datasets"


def val_metrics(model, x_test, y_test):
    print("Predicting on last 10%")
    yh = model.predict(x_test)
    print(yh.shape)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
    return {
        "mse": mse,
        "r2": r2,
    }


def rollout(cfg: Config, pi, venv, t: Timer):
    _ = venv.reset()

    frames = []
    total_success, done = 0, False

    while not done and len(frames) < cfg.steps:
        states = np.array(venv.get_sim_state())

        print(f"Predicting actions across {states.shape[0]} envs")
        with t("fwd"):
            actions = pi(states)
        with t("sim"):
            obs, rewards, dones, _info = venv.step(actions)

        done = np.array(dones).all()
        rewards = np.array(rewards)
        frames.append(obs[0]["galleryview_image"][::-1])

        successes = rewards.sum(axis=-1)  # BUG
        avg_sr = successes.mean()
        total_success += avg_sr
        print(f"Avg success rate: {avg_sr}")

        # step() sets done=self._check_success()
        cfg.wandb.log({"Gripper Pred": actions[0][:, -1]})

    # BUG
    avg_sr = total_success / len(frames)
    return {
        "video": wandb.Video(frames, fps=30, format="mp4"),
        "len": len(frames),
        "avg_sr": avg_sr,
    }


class ModelPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, states):
        return self.model.predict(states)


def main(cfg: Config):
    venv = cfg.env.build()

    print(data_dir)
    check_download(data_dir, cfg.task_suite)

    raw_data: dict[str, Any] = cfg.env.load_data(data_dir)
    features, actions = extract(raw_data, steps=True)

    # TODO refactor
    print("Globally Shuffled")
    rng = np.random.default_rng(seed=42)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    features = features[indices]
    actions = actions[indices]
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

    t = Timer()

    print(f"Fitting on {cfg.training * 100}%")
    with t("fit"):
        model.fit(x_fit, y_fit)

    with t("val"):
        val = val_metrics(model, x_test, y_test)

    print("Initializing Wandb")
    run = cfg.wandb.initialize(cfg)

    pi = ModelPolicy(model)
    result = rollout(cfg, pi, venv)

    # Save video
    # cfg.env.task.name

    times = t.get_average_times()
    metrics = {"val": val, "rollout": result, "times": times}
    cfg.wandb.log(metrics)

    venv.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
