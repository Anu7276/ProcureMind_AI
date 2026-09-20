"""
Unit tests for security middleware, health probes, and operational hardening (Phase R7).
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.config.settings import settings
from backend.middleware.security import _client_ip


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_knowledge():
    from ai.knowledge import knowledge_loader as kl
    kl.load_all()


def test_healthz_liveness_probe():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}

    resp_api = client.get("/api/healthz")
    assert resp_api.status_code == 200
    assert resp_api.json() == {"status": "alive"}


def test_readyz_readiness_probe():
    resp = client.get("/readyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ready"
    assert data.get("standards_loaded", 0) > 0


def test_trusted_proxy_client_ip_extraction():
    # 1. Trusted proxy (127.0.0.1) -> trusts X-Forwarded-For
    req_trusted = MagicMock()
    req_trusted.client.host = "127.0.0.1"
    req_trusted.headers = {"x-forwarded-for": "203.0.113.50, 10.0.0.1"}
    ip = _client_ip(req_trusted)
    assert ip == "203.0.113.50"

    # 2. Untrusted proxy (198.51.100.5) -> rejects spoofed X-Forwarded-For
    req_untrusted = MagicMock()
    req_untrusted.client.host = "198.51.100.5"
    req_untrusted.headers = {"x-forwarded-for": "1.2.3.4"}
    ip_untrusted = _client_ip(req_untrusted)
    assert ip_untrusted == "198.51.100.5"


def test_api_key_authentication(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret_test_key_123")

    # Open path /healthz should still pass without API key
    open_resp = client.get("/healthz")
    assert open_resp.status_code == 200

    # Protected path /recommend should return 401 without API key
    unauth_resp = client.post("/recommend", json={"raw_query": "Cement"})
    assert unauth_resp.status_code == 401
    assert unauth_resp.json().get("error") == "unauthorized"

    # Protected path with correct X-API-Key header should succeed
    auth_resp = client.post(
        "/recommend",
        json={"raw_query": "OPC 43 Grade Cement"},
        headers={"X-API-Key": "secret_test_key_123"},
    )
    assert auth_resp.status_code == 200
