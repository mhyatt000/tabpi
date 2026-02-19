from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import imageio
import libero
import numpy as np
from rich import print
import tyro

from tabpi.utils.util import check_download, EnvFactory, LiberoFactory
from tabpi.wab import Wandb
import wandb


@dataclass
class Config:
    task_suite: str = "libero_object"
    task_id: int = 0
    steps: int = 400
    training: float = 1

    wandb: Wandb = field(default_factory=Wandb)
    env: EnvFactory = field(default_factory=LiberoFactory)


data_dir = Path(libero.__file__).parents[0] / "datasets"


def main(cfg: Config):
    demo_path = cfg.env.get_data_path(data_dir)
    print(demo_path)

    with h5py.File(demo_path, "r") as f:
        states = f["data/demo_0/states"][()]
        actions = f["data/demo_0/actions"][()]
        init_state = states[0]

    # Needs to be same num of rows as env.n_envs
    init_states = np.stack([init_state] * 4)
    print(f"Inits shape: {init_states.shape}")
    venv = cfg.env.build()
    _ = venv.reset()
    obs = venv.set_init_state(init_states)

    print(data_dir)
    check_download(data_dir, cfg.task_suite)

    print("Initializing Wandb")
    run = cfg.wandb.initialize(cfg)

    frames = []
    total_time, total_success = 0, 0
    done_global, steps, max_steps = False, 0, cfg.steps

    vid_path = "ObsVids/"
    dir_path = Path(vid_path)
    dir_path.mkdir(exist_ok=True)

    while not done_global and steps < actions.shape[0]:
        print(f"Steps={steps}")

        print(f"Repeating demo_0 actions{steps}")
        action = np.stack([actions[steps]] * 4)
        obs, venv_rewards, done, _info = venv.step(action)
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

        cfg.wandb.log({"Success Rate": avg_sr})

        steps += 1

    avg_time = total_time / steps
    avg_sr = total_success / steps

    venv.close()

    # Save video
    task_name = cfg.env.get_benchmark(cfg.task_suite).get_task(cfg.task_id).name
    imageio.mimsave(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4", frames, fps=30)
    print(f"{vid_path}{int(cfg.training * 100)}%{task_name}All.mp4 saved")

    cfg.wandb.log(
        {
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
