"""
EcoML — API Scaffolding & Health Check Tests.

Verifies that the FastAPI application initializes correctly, reuses core settings,
and serves the /api/health endpoint with expected metadata.
"""

from fastapi.testclient import TestClient
from backend.main import app
from config.settings import PROJECT_NAME, PROJECT_VERSION, PROJECT_STATUS

client = TestClient(app)


def test_api_health_endpoint():
    """Verify that GET /api/health returns HTTP 200 with accurate EcoML metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == PROJECT_NAME
    assert data["version"] == PROJECT_VERSION
    assert data["operational_status"] == PROJECT_STATUS
    assert "tagline" in data

