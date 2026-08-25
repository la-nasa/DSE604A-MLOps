from src.data_processing import (
    SENSOR_FEATURE_COLUMNS,
    SENSOR_TARGET_COLUMN,
    generate_sensor_data,
    get_sensor_train_test_split,
)


def test_generate_sensor_data_shape_and_columns():
    df = generate_sensor_data(n_samples=500, random_state=0)
    assert len(df) == 500
    for col in SENSOR_FEATURE_COLUMNS + [SENSOR_TARGET_COLUMN]:
        assert col in df.columns


def test_generate_sensor_data_is_binary_and_imbalanced_realistically():
    df = generate_sensor_data(n_samples=2000, random_state=0)
    assert set(df["failure"].unique()) <= {0, 1}
    failure_rate = df["failure"].mean()
    # Predictive-maintenance data is imbalanced but not degenerate.
    assert 0.01 < failure_rate < 0.5


def test_get_sensor_train_test_split_shapes():
    X_train, X_test, y_train, y_test = get_sensor_train_test_split(test_size=0.25)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.25) < 0.02
    assert list(X_train.columns) == SENSOR_FEATURE_COLUMNS
