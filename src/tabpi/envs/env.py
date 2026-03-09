from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

Env: TypeAlias = Any


@dataclass
class EnvFactory:
    @abstractmethod
    def build(self) -> Env:
        pass

    @abstractmethod
    def load_data(self) -> Env:
        pass

    @abstractmethod
    def check_download(self) -> Env:
        pass

    @abstractmethod
    def set_init_state(self, init_state):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass
