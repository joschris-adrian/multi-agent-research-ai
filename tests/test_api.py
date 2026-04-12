import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

FAKE_RESULT = {
    "question": "What are AI trends?",
    "tasks": "1. Search\n2. Analyse",
    "documents": [{"title": "AI", "content": "AI is growing.", "source": "http://example.com"}],
    "insights": "AI is expanding across all sectors.",
    "report": "# AI Trends\n## Introduction\nAI is transforming industries.",
    "critic_feedback": "Clear and well structured."
}


def test_root():
    assert client.get("/").status_code == 200


@patch("api.main.system.run")
def test_research_returns_200(mock_run):
    mock_run.return_value = FAKE_RESULT
    r = client.post("/research", json={"query": "AI trends"})
    assert r.status_code == 200


@patch("api.main.system.run")
def test_research_response_has_expected_keys(mock_run):
    mock_run.return_value = FAKE_RESULT
    data = client.post("/research", json={"query": "AI trends"}).json()
    for key in ["question", "report", "tasks", "insights", "critic_feedback"]:
        assert key in data


def test_research_missing_query_returns_422():
    assert client.post("/research", json={}).status_code == 422


@patch("api.main.system.run")
def test_research_empty_query(mock_run):
    mock_run.return_value = FAKE_RESULT
    assert client.post("/research", json={"query": ""}).status_code == 200


def test_api_url_env_var():
    with patch.dict(os.environ, {"API_URL": "http://api:8000"}):
        assert os.getenv("API_URL") == "http://api:8000"
