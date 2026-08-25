import os
import shutil
import sys
import tempfile

# Ensure the project root is on sys.path so `src` is importable during tests,
# regardless of the working directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def isolated_mlflow_tracking_uri():
    """Point MLflow at a throwaway store for the whole test session.

    Without this, running the test suite creates real experiments, runs and
    registered models (e.g. "CI_RollbackTest_Model") inside the student's
    actual ./mlruns / mlflow.db, cluttering their real MLflow UI every time
    CI or `pytest` runs locally.
    """
    import mlflow

    tmp_dir = tempfile.mkdtemp(prefix="mlflow_test_")
    tracking_uri = f"sqlite:///{os.path.join(tmp_dir, 'mlflow_test.db')}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield tracking_uri
    finally:
        # On Windows, SQLAlchemy/sqlite can keep a file handle open past the
        # end of the test session, which makes an immediate rmtree fail with
        # PermissionError. Best-effort cleanup only; a stray temp dir under
        # %TEMP% is harmless and gets swept up by normal OS cleanup.
        shutil.rmtree(tmp_dir, ignore_errors=True)
