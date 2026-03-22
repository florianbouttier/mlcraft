"""Cross-validation splitters implemented without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.split.base import BaseCVSplitter
from mlcraft.utils.random import normalize_random_state


@dataclass
class KFoldSplitter:
    """Split data into `k` shuffled folds without scikit-learn.

    Args:
        n_splits: Number of folds to generate.
        shuffle: Whether to shuffle indices before partitioning them.
        random_state: Optional random seed or generator.
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | np.random.Generator | None = None

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation indices for each fold.

        Args:
            X: Feature data used to determine the number of rows.
            y: Unused target array accepted for interface compatibility.

        Returns:
            Iterator[tuple[np.ndarray, np.ndarray]]: Iterator yielding sorted
            train and validation index arrays.

        Raises:
            ValueError: If `n_splits` is invalid for the number of samples.
        """

        n_samples = len(X) if isinstance(X, np.ndarray) else len(next(iter(X.values())))
        if self.n_splits < 2 or self.n_splits > n_samples:
            raise ValueError("n_splits must be between 2 and n_samples.")
        indices = np.arange(n_samples)
        if self.shuffle:
            normalize_random_state(self.random_state).shuffle(indices)
        folds = np.array_split(indices, self.n_splits)
        for fold_idx in range(self.n_splits):
            val_idx = np.sort(folds[fold_idx])
            train_parts = [folds[idx] for idx in range(self.n_splits) if idx != fold_idx]
            train_idx = np.sort(np.concatenate(train_parts))
            yield train_idx, val_idx


@dataclass
class StratifiedKFoldSplitter:
    """Split binary classification data while preserving class balance.

    Args:
        n_splits: Number of folds to generate.
        shuffle: Whether to shuffle samples within each class before
            assignment.
        random_state: Optional random seed or generator.
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | np.random.Generator | None = None

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield stratified train and validation indices.

        Args:
            X: Feature data used only to determine the number of rows.
            y: Binary target array used for stratification.

        Returns:
            Iterator[tuple[np.ndarray, np.ndarray]]: Iterator yielding sorted
            train and validation index arrays.

        Raises:
            ValueError: If `y` is missing or not binary.
        """

        if y is None:
            raise ValueError("StratifiedKFoldSplitter requires y.")
        y_array = np.asarray(y)
        unique = np.unique(y_array)
        if unique.shape[0] != 2:
            raise ValueError("StratifiedKFoldSplitter currently supports binary classification only.")
        rng = normalize_random_state(self.random_state)
        fold_buckets: list[list[int]] = [[] for _ in range(self.n_splits)]
        for label in unique.tolist():
            label_indices = np.flatnonzero(y_array == label)
            if self.shuffle:
                rng.shuffle(label_indices)
            for idx, sample_index in enumerate(label_indices.tolist()):
                fold_buckets[idx % self.n_splits].append(sample_index)
        all_indices = np.arange(y_array.shape[0])
        for fold_indices in fold_buckets:
            val_idx = np.sort(np.asarray(fold_indices, dtype=int))
            train_idx = np.sort(np.setdiff1d(all_indices, val_idx, assume_unique=False))
            yield train_idx, val_idx


def resolve_cv_splitter(
    cv: int = 5,
    *,
    cv_splitter: BaseCVSplitter | None = None,
    task_spec: TaskSpec | None = None,
    shuffle: bool = True,
    random_state=None,
) -> BaseCVSplitter:
    """Resolve a cross-validation splitter from public options.

    Args:
        cv: Number of folds to use when no custom splitter is supplied.
        cv_splitter: Optional custom splitter implementing `split`.
        task_spec: Optional task specification used to pick a sensible
            default splitter.
        shuffle: Whether built-in splitters should shuffle before splitting.
        random_state: Optional random seed or generator for built-in
            splitters.

    Returns:
        BaseCVSplitter: Custom splitter when provided, otherwise a built-in
        k-fold or stratified k-fold splitter.
    """

    if cv_splitter is not None:
        return cv_splitter
    if task_spec is not None and task_spec.task_type == TaskType.CLASSIFICATION:
        return StratifiedKFoldSplitter(n_splits=cv, shuffle=shuffle, random_state=random_state)
    return KFoldSplitter(n_splits=cv, shuffle=shuffle, random_state=random_state)
