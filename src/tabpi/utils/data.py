"""Data loading and preprocessing (HDF5, demonstrations)."""

from __future__ import annotations

import os

import h5py
from libero.libero.utils.download_utils import libero_dataset_download
import numpy as np


def check_download(suites, task_suite_name):
    suite_dir = suites.joinpath(task_suite_name)

    if os.path.exists(suite_dir):
        print("Datasets found:")
        t_names = [f.stem for f in suite_dir.glob("*.hdf5")]
        for index, name in enumerate(t_names):
            print(index, ": ", name)
    else:
        print(f"{task_suite_name} datasets not found. Downloading now")
        libero_dataset_download(datasets=task_suite_name, download_dir=suites, use_huggingface=True)


def h5_to_tree(path: str) -> dict[str, h5py.Group]:
    def read_node(node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        if isinstance(node, h5py.Group):
            return {k: read_node(node[k]) for k in node}
        raise TypeError(type(node))

    with h5py.File(path, "r") as f:
        return read_node(f)


def extract(tree: dict, episodes: bool = False, steps: bool = False, demo=None) -> tuple[np.ndarray, np.ndarray]:
    demos = tree["data"]

    if demo is None:
        sa_by_demo = {k: (d["states"], d["actions"]) for k, d in demos.items()}

    else:
        key = f"demo_{demo}"
        print(key)
        sa_by_demo = {key: (demos[key]["states"], demos[key]["actions"])}

    keys = sa_by_demo.keys()

    if steps:
        states = np.concatenate([sa_by_demo[k][0] for k in keys], axis=0)
        actions = np.concatenate([sa_by_demo[k][1] for k in keys], axis=0)
    elif episodes:
        states = np.array([sa_by_demo[k][0] for k in keys], dtype=object)
        actions = np.array([sa_by_demo[k][1] for k in keys], dtype=object)

    return states, actions


# Assume test param is for the percentage of tail end of data you want to test on
def split(
    training: float, test: float, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(x.shape[0] * training)
    n_test = int(x.shape[0] * test)

    x_train, x_test = x[:n_train], x[-n_test:]
    y_train, y_test = y[:n_train], y[-n_test:]

    return x_train, x_test, y_train, y_test
