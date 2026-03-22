import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.split.cv import KFoldSplitter, StratifiedKFoldSplitter, resolve_cv_splitter


def test_kfold_is_reproducible():
    splitter_a = KFoldSplitter(n_splits=3, random_state=42)
    splitter_b = KFoldSplitter(n_splits=3, random_state=42)
    splits_a = list(splitter_a.split(np.arange(9)))
    splits_b = list(splitter_b.split(np.arange(9)))
    assert [val.tolist() for _, val in splits_a] == [val.tolist() for _, val in splits_b]


def test_stratified_kfold_preserves_label_balance(classification_data):
    X, y = classification_data
    splitter = StratifiedKFoldSplitter(n_splits=4, random_state=12)
    for _, val_idx in splitter.split(X, y):
        assert y[val_idx].sum() == 1


def test_resolve_cv_splitter_uses_stratification_for_classification():
    splitter = resolve_cv_splitter(4, task_spec=TaskSpec(task_type="classification"), random_state=3)
    assert isinstance(splitter, StratifiedKFoldSplitter)

