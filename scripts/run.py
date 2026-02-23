from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import libero
import numpy as np
from rich import print
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
import tyro
import wandb

from tabpi.envs.libero import EnvFactory, LiberoFactory
from tabpi.models.tabpfn_policy import ModelPolicy
from tabpi.utils.data import check_download, extract, split
from tabpi.utils.eval import rollout, val_metrics
from tabpi.utils.timer import Timer
from tabpi.utils.wab import Wandb


@dataclass
class Config:
    task_suite: str = "libero_object"
    task_id: int = 0
    steps: int = 400
    training: float = 0.10

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)
    debug: bool = False

    def __post_init__(self):
        if self.debug:
            self.wandb.use = False


data_dir = Path(libero.__file__).parents[0] / "datasets"


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
    result = rollout(cfg.steps, pi, venv, t)

    # Save video
    # cfg.env.task.name

    times = t.get_average_times()
    metrics = {"val": val, "rollout": result, "times": times}
    print(metrics)
    cfg.wandb.log(metrics)

    venv.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
