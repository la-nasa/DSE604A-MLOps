import pytest
import sys
import os

# ensure project root is on sys.path so `src` package is importable during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train_register


def test_train_and_register_runs():
    # run with small forest and low threshold to avoid registry promotion
    res = train_register.train_and_register(n_estimators=10, max_depth=2, registered_model_name="TestModel", stage_threshold=0.0, experiment_name="CI_Test")
    assert "run_id" in res
    assert "metrics" in res
    m = res["metrics"]
    assert isinstance(m.get("accuracy"), float)
