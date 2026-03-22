from mlcraft.metrics.registry import default_metric_registry


def test_metric_registry_mapping_and_score_direction():
    rmse = default_metric_registry.get("rmse")
    auc = default_metric_registry.get("roc_auc")
    assert rmse.backend_names["xgboost"] == "rmse"
    assert auc.backend_names["catboost"] == "AUC"
    assert rmse.to_score(2.0) == -2.0
    assert auc.to_score(0.8) == 0.8

