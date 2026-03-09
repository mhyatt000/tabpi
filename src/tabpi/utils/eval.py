"""Reusable evaluation helpers for model validation and environment rollout."""

from __future__ import annotations

import random
from typing import Any

import imageio
import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from tabpi.envs.env import EnvFactory
import wandb


def val_metrics(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    print("Predicting on last 10%")
    yh = model.predict(x_test)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    return {"mse": mse, "r2": r2}


def rollout(
    env: EnvFactory,
    max_steps: int,
    policy: Any,
    venv: Any,
    timer: Any,
    overfit: bool = False,
    demo: bool = False,
    init_state=None,
) -> dict[str, Any]:
    env_name = "demo" if demo else "sim"
    if overfit:
        print("Overfitting")
        env.set_init_state(init_state)

    frames = []
    success = 0

    bar = tqdm(range(max_steps), desc="Rollout")

    for i in bar:
        states = np.array(env.get_state())

        with timer("fwd"):
            actions = policy(states) if not isinstance(policy, np.ndarray) else np.stack([policy[i]] * env.n_envs)
        with timer(env_name):
            obs, rewards, dones, _info = env.step(actions)

        # frames.append(obs[0]["galleryview_image"][::-1])
        frames.append(obs["agentview_image"][::-1])

        dones = dones  # np.array(dones)
        rewards = rewards  # np.array(rewards)
        successes = rewards  # rewards.sum(axis=-1)

        desc = f"Step: {len(frames)}/{max_steps} SR: {successes: .2f} Done: {dones}"
        bar.set_description(desc)

        if dones:  # dones.all():
            bar.write("Task Completed!")
            break

    env.reset()

    imageio.mimsave(f"ObsVids/{env_name}_rollout{random.randint(1, 1000)}.mp4", frames, fps=30)

    return {
        f"{env_name}/video": wandb.Video(f"ObsVids/{env_name}_rollout.mp4", format="mp4"),
        "len": len(frames),
        "sr": successes,
    }
