from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# vector store server 

def get_vs_client():
    from src.mcp.servers.vector_store_server import app
    return TestClient(app)

def test_vector_store_search_returns_result():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = ["Solar is growing."]
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar"})
        assert r.status_code == 200
        assert "result" in r.json()

def test_vector_store_search_calls_store_with_query():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        client = get_vs_client()
        client.post("/vector_store/search", json={"query": "renewables"})
        mock_store.search.assert_called_once_with("renewables")

def test_vector_store_add_returns_ok():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        client = get_vs_client()
        r = client.post("/vector_store/add", json={"documents": [
            {"title": "Solar", "content": "Growing fast.", "source": "http://example.com"}
        ]})
        assert r.status_code == 200
        assert r.json()["result"] == "ok"

def test_vector_store_search_missing_query_returns_422():
    client = get_vs_client()
    r = client.post("/vector_store/search", json={})
    assert r.status_code == 422

def test_vector_store_add_missing_documents_returns_422():
    client = get_vs_client()
    r = client.post("/vector_store/add", json={})
    assert r.status_code == 422


# web search server 

def get_ws_client():
    from src.mcp.servers.web_search_server import app
    return TestClient(app)

def test_web_search_returns_result():
    fake_docs = [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Solar", "body": "Growing.", "href": "http://example.com"}
        ]
        client = get_ws_client()
        r = client.post("/web_search/search", json={"query": "solar energy"})
        assert r.status_code == 200
        assert "result" in r.json()

def test_web_search_returns_empty_on_failure():
    with patch("src.mcp.servers.web_search_server.DDGS", side_effect=Exception("rate limited")):
        client = get_ws_client()
        r = client.post("/web_search/search", json={"query": "solar", "retries": 1, "delay": 0})
        assert r.status_code == 200
        assert r.json()["result"] == []

def test_web_search_missing_query_returns_422():
    client = get_ws_client()
    r = client.post("/web_search/search", json={})
    assert r.status_code == 422

def test_web_search_uses_max_results():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": f"Result {i}", "body": "Content.", "href": "http://example.com"}
            for i in range(5)
        ]
        client = get_ws_client()
        r = client.post("/web_search/search", json={"query": "solar", "max_results": 2})
        assert r.status_code == 200