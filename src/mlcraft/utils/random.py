"""Random state normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_random_state(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a numpy Generator from a supported random state input."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def make_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Alias for normalize_random_state."""

    return normalize_random_state(random_state)

