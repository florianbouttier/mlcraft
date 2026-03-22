"""Base splitter protocol."""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class BaseCVSplitter(Protocol):
    """Define the protocol expected from cross-validation splitters."""

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation indices.

        Args:
            X: Feature data used only to determine the number of rows.
            y: Optional target array required by some splitters.

        Returns:
            Iterator[tuple[np.ndarray, np.ndarray]]: Iterator yielding
            `(train_idx, val_idx)` arrays.
        """
