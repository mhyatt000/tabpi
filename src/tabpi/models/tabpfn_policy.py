"""TabPFN-based policy wrapper/multi-output handler."""

from __future__ import annotations

from typing import Any


class ModelPolicy:
    """Thin policy wrapper around models exposing ``predict``."""

    def __init__(self, model: Any):
        self.model = model

    def __call__(self, states: Any) -> Any:
        return self.model.predict(states)
