from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import libero
from rich import print
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
import tyro
import wandb

from tabpi.envs.libero import EnvFactory, LiberoFactory
from tabpi.models.tabpfn_policy import ModelPolicy
from tabpi.utils.data import check_download, extract, shuffle
from tabpi.utils.eval import rollout, val_metrics
from tabpi.utils.timer import Timer
from tabpi.utils.wab import Wandb


@dataclass
class Config:
    fit: float = 0.10
    shuffle: str = None

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)
    debug: bool = False

    def __post_init__(self):
        if self.debug:
            self.wandb.use = False


data_dir = Path(libero.__file__).parents[0] / "datasets"


def main(cfg: Config):
    venv = cfg.env.build()

    check_download(data_dir, cfg.env.suite)

    raw_data: dict[str, Any] = cfg.env.load_data(data_dir)
    features, actions = extract(raw_data, cfg.shuffle, cfg.env.demo)

    x_fit, x_test, y_fit, y_test = shuffle(cfg.fit, features, actions, cfg.shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        n_estimators=6,
        device=device,
        # device='cuda:0',
        # n_preprocessing_jobs=act_dim*2,
        # memory_saving_mode=False,
        inference_precision="autocast",
    )

    t = Timer()

    print(f"Fitting on {cfg.fit * 100}%")
    with t("fit"):
        model.fit(x_fit, y_fit)

    with t("val"):
        val = val_metrics(model, x_test, y_test)

    print("Initializing Wandb")
    cfg.wandb.initialize(cfg)

    if cfg.env.overfit:
        print(f"Running demo_{cfg.env.demo}")
        demo_result = rollout(cfg.env.horizon, actions, venv, t, cfg.env.overfit, features[0])

    # TODO use this in rollout, instead of passing it in
    print(f"Horizon: {venv.get_env_attr('horizon')}")
    pi = ModelPolicy(model)
    result = rollout(cfg.env.horizon, pi, venv, t)

    times = t.get_average_times()
    metrics = {
        "val": val,
        **({"demo_rollout": demo_result} if cfg.env.overfit else {}),
        "rollout": result,
        "times": times,
    }
    print(metrics)
    cfg.wandb.log(metrics)

    venv.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
