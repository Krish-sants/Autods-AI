import io

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_gate_disabled_by_default_allows_requests(client):
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200


def test_gate_blocks_requests_without_password(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ACCESS_PASSWORD", "secret123")
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 401


def test_gate_blocks_requests_with_wrong_password(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ACCESS_PASSWORD", "secret123")
    resp = client.get("/api/v1/runs", headers={"X-App-Password": "wrong"})
    assert resp.status_code == 401


def test_gate_allows_requests_with_correct_password(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ACCESS_PASSWORD", "secret123")
    resp = client.get("/api/v1/runs", headers={"X-App-Password": "secret123"})
    assert resp.status_code == 200


def test_health_endpoint_never_requires_password(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ACCESS_PASSWORD", "secret123")
    resp = client.get("/health")
    assert resp.status_code == 200


def test_upload_blocked_without_password(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "APP_ACCESS_PASSWORD", "secret123")
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert resp.status_code == 401
