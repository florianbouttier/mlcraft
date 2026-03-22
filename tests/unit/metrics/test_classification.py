import numpy as np

from mlcraft.metrics import classification


def test_classification_metrics_on_easy_case():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = (y_score >= 0.5).astype(int)
    assert classification.roc_auc(y_true, y_score=y_score) == 1.0
    assert classification.pr_auc(y_true, y_score=y_score) == 1.0
    assert classification.accuracy(y_true, y_pred=y_pred) == 1.0
    assert classification.f1(y_true, y_pred=y_pred) == 1.0
    assert classification.gini(y_true, y_score=y_score) == 1.0


def test_classification_metrics_support_weights():
    y_true = np.array([0, 1, 1])
    y_score = np.array([0.2, 0.8, 0.6])
    weights = np.array([1.0, 2.0, 1.0])
    assert classification.logloss(y_true, y_score=y_score, sample_weight=weights) > 0

