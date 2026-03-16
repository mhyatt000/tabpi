from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from rich import print
from tabpfn_extensions.multioutput import TabPFNMultiOutputRegressor
import torch
from tqdm import tqdm
import tyro
import wandb

from tabpi.envs.env import EnvFactory
from tabpi.envs.robosuite import RoboSuiteFactory
from tabpi.models.tabpfn_policy import ModelPolicy
from tabpi.utils.data import extract, shuffle
from tabpi.utils.eval import rollout, val_metrics
from tabpi.utils.timer import Timer
from tabpi.utils.wab import Wandb


@dataclass
class Config:
    fit: float = 0.10
    selection: str = "steps"
    n_runs: int = 5  # for non VecEnv. sequential rollouts

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=RoboSuiteFactory)
    debug: bool = False

    action: Literal["absolute", "relative"] = "relative"
    n_estimators: int = 4

    def __post_init__(self):
        if self.debug:
            self.wandb.use = False


def main(cfg: Config):
    venv = cfg.env.build()

    cfg.env.check_download()

    raw_data: dict[str, Any] = cfg.env.load_data()
    features, actions = extract(raw_data, cfg.selection, cfg.env.demo)
    print(features.shape, actions.shape)

    x_fit, x_test, y_fit, y_test = shuffle(cfg.fit, features, actions, None)

    print(x_fit.shape, y_fit.shape, x_test.shape, y_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNMultiOutputRegressor(
        n_estimators=cfg.n_estimators,
        device=device,
        # device='cuda:0',:
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
        demo_result = rollout(cfg.env, cfg.env.horizon, actions, venv, t, cfg.env.overfit, True, features[0])

    print(f"Horizon: {cfg.env.horizon}")
    pi = ModelPolicy(model)

    # results : list[dict[str, Any]] = [roll(n) for n in range(cfg.n_runs)] # if cfg.n_runs != 1 else roll()

    roll = partial(rollout, cfg.env, cfg.env.horizon, pi, venv, t, cfg.wandb, cfg.env.overfit, False, features[0])
    for i in tqdm(range(cfg.n_runs), desc="Rollouts", leave=False):
        result = roll()
        print(result)
        cfg.wandb.log(result)

    times = t.get_average_times()

    """
    # if cfg.n_runs != 1: # then we want to log each video separately
    for i, r in enumerate(results):
        for key in ["demo/video", "sim/video"]:
            if key in r:
                cfg.wandb.log({f"{key}{i}": r.pop(key)})

# reduce list by taking the average of each metric across runs with jax.tree.map and/or jax.tree.reduce
    results = jax.tree.reduce(lambda x, y: jax.tree.map(lambda a, b: a + b, x, y), results) / cfg.n_runs
    """

    metrics = {
        "val": val,
        **({"demo_rollout": demo_result} if cfg.env.overfit else {}),
        # "rollout": results,
        "times": times,
    }
    print(metrics)
    for _ in range(2):  # hack: turn bars to lines
        cfg.wandb.log(metrics)

    venv.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
