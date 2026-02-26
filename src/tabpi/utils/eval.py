"""Reusable evaluation helpers for model validation and environment rollout."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich import print
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

import wandb


def val_metrics(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    print("Predicting on last 10%")
    yh = model.predict(x_test)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    return {"mse": mse, "r2": r2}


def rollout(max_steps: int, policy: Any, venv: Any, timer: Any, demo: bool = False, init_state=None) -> dict[str, Any]:
    if demo:
        env_name = "demo"
        init_states = np.stack([init_state] * venv.env_num)
        venv.set_init_state(init_states)
    else:
        env_name = "sim"

    frames = []
    success = 0

    bar = tqdm(range(max_steps), desc="Rollout")

    for i in bar:
        states = np.array(venv.get_sim_state())

        with timer("fwd"):
            actions = policy(states) if not isinstance(policy, np.ndarray) else np.stack([policy[i]] * venv.env_num)
        with timer(env_name):
            obs, rewards, dones, _info = venv.step(actions)

        frames.append(obs[0]["galleryview_image"][::-1])

        dones = np.array(dones)
        rewards = np.array(rewards)
        successes = rewards.sum(axis=-1)

        desc = f"Step: {len(frames)}/{max_steps} SR: {successes} Done: {dones}"
        bar.set_description(desc)

        if dones.all():
            bar.write("Task Completed!")
            break

    venv.reset()

    return {
        f"{env_name}/video": wandb.Video(np.array(frames), fps=30, format="mp4"),
        "len": len(frames),
        "sr": successes,
    }
