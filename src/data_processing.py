"""
Data loading and preprocessing utilities.

Covers both the generic CSV loader used by the earlier labs and the
Lab 15 capstone dataset: an industrial sensor dataset (temperature,
vibration, pressure, operating hours) used to predict machine failure.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SENSOR_FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "operating_hours"]
SENSOR_TARGET_COLUMN = "failure"

DEFAULT_SENSOR_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sensor_data.csv"
)


def load_data(path: str = None, target_column: str = None):
    """Load a generic dataset. If path is None, returns (None, None) so the
    caller can fall back to a default dataset.

    Returns X, y
    """
    if path is None:
        return None, None

    df = pd.read_csv(path)
    if target_column is None:
        raise ValueError("target_column must be provided when loading from CSV")

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_sensor_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic but physically-motivated industrial sensor dataset.

    Simulates readings from machines fitted with temperature, vibration and
    pressure sensors, plus cumulative operating hours. Failure risk increases
    with higher temperature, higher vibration, pressure deviating from the
    nominal operating band, and more accumulated operating hours -- mirroring
    the Lab 15 capstone scenario (predictive maintenance for manufacturing
    equipment).
    """
    rng = np.random.default_rng(random_state)

    temperature = rng.normal(loc=70, scale=10, size=n_samples).clip(30, 130)
    vibration = rng.gamma(shape=2.0, scale=1.2, size=n_samples)
    pressure = rng.normal(loc=100, scale=8, size=n_samples).clip(60, 150)
    operating_hours = rng.uniform(0, 20000, size=n_samples)

    # Standardize the drivers of failure risk to build an interpretable logit.
    z_temp = (temperature - 70) / 10
    z_vib = (vibration - vibration.mean()) / vibration.std()
    z_pressure_dev = (np.abs(pressure - 100) - 8) / 8
    z_hours = (operating_hours - operating_hours.mean()) / operating_hours.std()

    logit = (
        -3.2
        + 1.1 * z_temp
        + 1.4 * z_vib
        + 0.7 * z_pressure_dev
        + 0.9 * z_hours
        + rng.normal(0, 0.5, size=n_samples)
    )
    failure_probability = 1 / (1 + np.exp(-logit))
    failure = (rng.uniform(0, 1, size=n_samples) < failure_probability).astype(int)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "operating_hours": operating_hours,
            "failure": failure,
        }
    )


def load_sensor_data(path: str = DEFAULT_SENSOR_DATA_PATH, generate_if_missing: bool = True) -> pd.DataFrame:
    """Load the Lab 15 capstone sensor dataset, generating it on first use."""
    if not os.path.exists(path):
        if not generate_if_missing:
            raise FileNotFoundError(f"Sensor dataset not found at {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = generate_sensor_data()
        df.to_csv(path, index=False)
        return df

    return pd.read_csv(path)


def get_sensor_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """Load the sensor dataset and return an (X_train, X_test, y_train, y_test) split."""
    df = load_sensor_data()
    X = df[SENSOR_FEATURE_COLUMNS]
    y = df[SENSOR_TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    dataset = load_sensor_data()
    print(f"Sensor dataset ready: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(dataset["failure"].value_counts(normalize=True))
