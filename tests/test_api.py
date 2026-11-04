"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from spectrarag.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_ingest_and_query_endpoints(client):
    ingest_payload = {
        "documents": [
            {
                "id": "doc_test_api",
                "text": "SpectraRAG integrates LangGraph multi-agent cyclic loops with Hybrid RRF retrieval.",
                "metadata": {"env": "test"}
            }
        ]
    }
    ingest_res = client.post("/api/v1/ingest", json=ingest_payload)
    assert ingest_res.status_code == 201
    assert ingest_res.json()["documents_ingested"] == 1

    query_payload = {
        "query": "What does SpectraRAG integrate for multi-agent loops?"
    }
    query_res = client.post("/api/v1/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert "latency_ms" in data
