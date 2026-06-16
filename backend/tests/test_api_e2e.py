import io
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Must be used as a context manager so FastAPI's lifespan (init_db) runs;
    # a bare TestClient(app) never triggers startup and "no such table: runs" results.
    with TestClient(app) as c:
        yield c


def _poll_until(client: TestClient, run_id: str, target_statuses: set[str], max_iters: int = 50) -> dict:
    status = {}
    for _ in range(max_iters):
        resp = client.get(f"/api/v1/runs/{run_id}/status")
        assert resp.status_code == 200
        status = resp.json()
        if status["status"] in target_statuses:
            return status
        time.sleep(0.05)
    raise TimeoutError(f"Run {run_id} did not reach {target_statuses}, last status: {status}")


def test_full_pipeline_classification_flow(client, synthetic_classification_df):
    csv_bytes = synthetic_classification_df.to_csv(index=False).encode("utf-8")

    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    upload_data = upload_resp.json()
    run_id = upload_data["run_id"]
    assert upload_data["format"] == "csv"
    assert upload_data["size_bytes"] == len(csv_bytes)

    start_resp = client.post(f"/api/v1/runs/{run_id}/start")
    assert start_resp.status_code == 200, start_resp.text

    status = _poll_until(client, run_id, {"awaiting_target_confirmation", "failed"})
    assert status["status"] == "awaiting_target_confirmation", status
    assert status["candidate_target"] == "churn"
    assert status["candidate_target_confidence"] > 0
    assert status["candidate_target_reasoning"]

    confirm_resp = client.post(f"/api/v1/runs/{run_id}/confirm-target", json={"target_column": "churn"})
    assert confirm_resp.status_code == 200, confirm_resp.text
    assert confirm_resp.json()["problem_type"] == "classification"

    status = _poll_until(client, run_id, {"complete", "failed"}, max_iters=100)
    assert status["status"] == "complete", status
    assert status["best_model_id"] is not None

    summary_resp = client.get(f"/api/v1/runs/{run_id}/summary")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["n_rows"] == len(synthetic_classification_df)

    eda_resp = client.get(f"/api/v1/runs/{run_id}/eda")
    assert eda_resp.status_code == 200
    assert "figures" in eda_resp.json()

    leaderboard_resp = client.get(f"/api/v1/runs/{run_id}/leaderboard")
    assert leaderboard_resp.status_code == 200
    leaderboard = leaderboard_resp.json()["leaderboard"]
    assert len(leaderboard) == 5
    best_model_id = leaderboard_resp.json()["best_model_id"]

    metrics_resp = client.get(f"/api/v1/runs/{run_id}/metrics/{best_model_id}")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    fi_resp = client.get(f"/api/v1/runs/{run_id}/feature-importance")
    assert fi_resp.status_code == 200

    shap_resp = client.get(f"/api/v1/runs/{run_id}/shap")
    assert shap_resp.status_code == 200
    assert shap_resp.json()["available"] is True

    report_resp = client.get(f"/api/v1/runs/{run_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert len(report["markdown"]) > 0
    assert len(report["html"]) > 0
    # No GOOGLE_API_KEY is configured in the test environment (see conftest.py),
    # so the executive summary must fall back to the deterministic template path.
    assert report["executive_summary_source"] == "template"

    model_download = client.get(f"/api/v1/runs/{run_id}/download/model")
    assert model_download.status_code == 200
    assert len(model_download.content) > 0

    report_download = client.get(f"/api/v1/runs/{run_id}/download/report-html")
    assert report_download.status_code == 200
    assert len(report_download.content) > 0


def test_full_pipeline_regression_flow(client, synthetic_regression_df):
    csv_bytes = synthetic_regression_df.to_csv(index=False).encode("utf-8")

    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("housing.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    run_id = upload_resp.json()["run_id"]

    client.post(f"/api/v1/runs/{run_id}/start")
    status = _poll_until(client, run_id, {"awaiting_target_confirmation", "failed"})
    assert status["status"] == "awaiting_target_confirmation"

    confirm_resp = client.post(f"/api/v1/runs/{run_id}/confirm-target", json={"target_column": "price"})
    assert confirm_resp.json()["problem_type"] == "regression"

    status = _poll_until(client, run_id, {"complete", "failed"}, max_iters=100)
    assert status["status"] == "complete", status

    metrics_resp = client.get(f"/api/v1/runs/{run_id}/metrics/{status['best_model_id']}")
    metrics = metrics_resp.json()
    assert "rmse" in metrics
    assert "r2" in metrics


def test_messy_csv_does_not_crash_cleaning_and_eda(client, messy_csv_text):
    upload_resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("messy.csv", io.BytesIO(messy_csv_text.encode("utf-8")), "text/csv")},
    )
    run_id = upload_resp.json()["run_id"]

    client.post(f"/api/v1/runs/{run_id}/start")
    status = _poll_until(client, run_id, {"awaiting_target_confirmation", "failed"})
    assert status["status"] == "awaiting_target_confirmation", status

    cleaning_resp = client.get(f"/api/v1/runs/{run_id}/summary")
    assert cleaning_resp.status_code == 200
