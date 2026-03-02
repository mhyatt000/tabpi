"""LIBERO environment integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import SubprocVectorEnv

from tabpi.utils.data import h5_to_tree

Env: TypeAlias = Any


@dataclass
class EnvFactory:
    pass


@dataclass
class LiberoFactory(EnvFactory):
    suite: str = "libero_object"
    id: int = 0
    n_envs: int = 4
    vectorized: bool = True

    # TODO Unhide this
    horizon: int = 400

    overfit: bool = False
    demo: int = None

    def __post_init__(self):
        self.bench = benchmark.get_benchmark(self.suite)()
        self.task = self.bench.get_task(self.id)

        if self.overfit and self.demo is None:
            self.demo = 0

    def build(self) -> Env:
        bddl_file_path: Path = self.bench.get_task_bddl_file_path(self.id)

        print(f"Using task: {self.task.name}")

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
            "horizon": self.horizon,  # max number of steps per episode
        }

        if self.vectorized:
            env_fns = [lambda: OffScreenRenderEnv(**env_args) for _ in range(self.n_envs)]
            return SubprocVectorEnv(env_fns)

        env = OffScreenRenderEnv(**env_args)
        return env

    def load_data(self, suites_root: Path):
        data_path = suites_root / self.bench.get_task_demonstration(self.id)

        tree = h5_to_tree(data_path)
        return tree
