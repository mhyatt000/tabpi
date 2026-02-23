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
    print(yh.shape)

    mse = mean_squared_error(y_test, yh)
    r2 = r2_score(y_test, yh)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
    return {"mse": mse, "r2": r2}


def rollout(max_steps: int, policy: Any, venv: Any, timer: Any) -> dict[str, Any]:
    _ = venv.reset()

    frames = []
    total_success = 0.0

    bar = tqdm(range(max_steps), desc="Rollout")

    for _ in bar:
        states = np.array(venv.get_sim_state())

        with timer("fwd"):
            actions = policy(states)
        with timer("sim"):
            obs, rewards, dones, _info = venv.step(actions)

        rewards = np.array(rewards)
        frames.append(obs[0]["galleryview_image"][::-1])

        successes = rewards.sum(axis=-1)
        avg_sr = successes.mean()
        total_success += avg_sr

        desc = f"Step: {len(frames)}/{max_steps}, Avg SR: {avg_sr:.3f}"
        bar.set_description(desc)

        if np.array(dones).all():
            break

    avg_sr = total_success / len(frames) if frames else 0.0
    return {
        "video": wandb.Video(frames, fps=30, format="mp4"),
        "len": len(frames),
        "avg_sr": avg_sr,
    }
