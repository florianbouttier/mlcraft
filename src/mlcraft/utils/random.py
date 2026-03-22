"""Random state normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_random_state(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a numpy generator from a supported random-state input.

    Args:
        random_state: Optional seed or existing generator.

    Returns:
        np.random.Generator: Normalized generator instance.
    """

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def make_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a numpy generator for the provided random-state input.

    Args:
        random_state: Optional seed or existing generator.

    Returns:
        np.random.Generator: Normalized generator instance.
    """

    return normalize_random_state(random_state)
