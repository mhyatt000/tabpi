"""LIBERO environment integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import SubprocVectorEnv
from libero.libero.utils.download_utils import libero_dataset_download
import numpy as np

from tabpi.utils.data import h5_to_tree

from .env import EnvFactory


@dataclass
class LiberoFactory(EnvFactory):
    suite: str = "libero_object"
    id: int = 0
    n_envs: int = 4
    vectorized: bool = True

    # TODO Unhide this
    horizon: int = 400

    overfit: bool = False
    demo: int | None = None

    def __post_init__(self):
        self.bench = benchmark.get_benchmark(self.suite)()
        self.task = self.bench.get_task(self.id)
        self.suites_path = Path(libero.__file__).parents[0] / "datasets" / self.bench.get_task_demonstration(self.id)

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
            self.env = SubprocVectorEnv(env_fns)
            return self.env

        self.env = OffScreenRenderEnv(**env_args)
        return self.env

    def load_data(self):
        tree = h5_to_tree(self.suites_path)
        return tree

    def check_download(self):
        if os.path.exists(self.suites_path):
            print("Datasets found:")
            t_names = [f.stem for f in self.suites_path.glob("*.hdf5")]
            for index, name in enumerate(t_names):
                print(index, ": ", name)
        else:
            print(f"{self.suite} datasets not found. Downloading now")
            libero_dataset_download(datasets=self.suite, download_dir=self.suites_path, use_huggingface=True)

    def set_init_state(self, init_state):
        if self.vectorized:
            init_state = np.stack([init_state] * self.n_envs)
        self.env.set_init_state(init_state)

    def get_state(self):
        state_data = self.env.get_sim_state()
        print(state_data.shape)
        return state_data

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
