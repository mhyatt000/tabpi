"""LIBERO environment integration."""

from __future__ import annotations

from dataclasses import dataclass
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
    suite: str = "libero_object"  # used to select group of envs
    id: int = 0
    max_steps: int = 400
    n_envs: int = 4
    vectorized: bool = True  # you can set this to False to test non-vectorized envs

    # TODO why does field resolve to tyro subcommand ?
    # task: str = field(init=False)  # used to search for dataset name

    def __post_init__(self):
        bench = self.get_benchmark(self.suite)
        self.task = bench.get_task(self.id)

    def get_benchmark(self, suite: str) -> benchmark.Benchmark:
        return benchmark.get_benchmark(self.suite)()

    def build(self) -> Env:
        bench = self.get_benchmark(self.suite)

        bddl_file_path: Path = bench.get_task_bddl_file_path(self.id)

        print(f"Using task: {self.task.name}")

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 720,  # HD resolution
            "camera_widths": 1280,
            "camera_names": "galleryview",
            # TODO max steps ...
        }

        if self.vectorized:
            env_fns = [lambda: OffScreenRenderEnv(**env_args) for _ in range(self.n_envs)]
            return SubprocVectorEnv(env_fns)

        env = OffScreenRenderEnv(**env_args)
        return env

    def get_data_path(self, suites_root: Path):
        bench = self.get_benchmark(self.suite)
        demo_path: str = bench.get_task_demonstration(self.id)
        full_path: Path = suites_root / demo_path
        return full_path

    def load_data(self, suites_root: Path):
        data_path = self.get_data_path(suites_root)
        tree = h5_to_tree(data_path)
        return tree
