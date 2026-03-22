from mlcraft.split.train_test import train_test_split_random, train_test_split_time


def test_random_train_test_split_is_reproducible(regression_data):
    X, y = regression_data
    split_a = train_test_split_random(X, y, test_size=0.33, random_state=7)
    split_b = train_test_split_random(X, y, test_size=0.33, random_state=7)
    assert split_a[2].tolist() == split_b[2].tolist()
    assert split_a[3].tolist() == split_b[3].tolist()


def test_random_train_test_split_sizes(regression_data):
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split_random(X, y, test_size=2, random_state=1)
    assert len(y_train) == 4
    assert len(y_test) == 2
    assert len(X_train["num_a"]) == 4


def test_time_split_keeps_most_recent_for_test(temporal_data):
    X, y = temporal_data
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="date", test_size=2)
    assert X_test["value"].tolist() == [4, 5]
    assert y_test.tolist() == [40, 50]
    assert X_train["value"].tolist() == [1, 2, 3]

