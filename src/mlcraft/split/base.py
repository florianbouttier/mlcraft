"""Base splitter protocol."""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class BaseCVSplitter(Protocol):
    """Protocol for CV splitters."""

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation indices."""

