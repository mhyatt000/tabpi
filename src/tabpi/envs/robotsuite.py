"""Robotsuite environment integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, TypeAlias

import robomimic
import robosuite as suite
from robosuite.controllers import load_controller_config

from tabpi.utils.data import h5_to_tree

Env: TypeAlias = Any


@dataclass
class RobotSuiteFactory(EnvFactory):
    task: str = "Lift"
    n_envs: int = 4
    vectorized: bool = True
    horizon: int = 600

    def build(self):
        controller_config = load_controller_config(default_controller="OSC_POSE")

        env_kwargs = {
            "env_name": self.task,
            "robots": "Panda",
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": True,
            "camera_names": "galleryview",
            "camera_heights": 720,
            "camera_widths": 1280,
            "control_freq": 20,
            "horizon": self.horizon,
            "reward_shaping": True,
        }

        if self.vectorized:
            env_fns = [lambda: suite.make(**env_kwargs) for _ in range(self.n_envs)]
            return SubprocVectorEnv(env_fns)

        env = suite.make(**env_kwargs)
        return env

    def load_data(self, data_path):
        self.suites_path = Path(robomimic.__file__).parents[0] / "../datasets"
        data_path = self.suites_path / self.task.lower()

        tree = h5_to_tree(data_path)
        return tree

    def check_download(self):
        if os.path.exists(self.suites_path):
            print("Datasets found:")
            t_names = [f.stem for f in self.suites_path.glob("*.hdf5")]
            for index, name in enumerate(t_names):
                print(index, ": ", name)
        else:
            print(f"{self.Task} dataset not found. Downloading now")
            cmd = [
                sys.executable,
                "../../../.venv/lib/python3.13/site-packages/datasets",
                "--tasks",
                self.task.lower(),
                "--dataset_types",
                "ph",
                "--hdf5_types",
                "low_dim",
            ]
            subprocess.run(cmd, check=True)
