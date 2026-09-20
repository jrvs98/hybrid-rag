from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_service_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "hybrid-rag"


def test_ready_reports_service_status() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["status"] == "ready"


def test_request_id_is_preserved_when_provided() -> None:
    response = client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.headers["X-Request-ID"] == "request-123"
