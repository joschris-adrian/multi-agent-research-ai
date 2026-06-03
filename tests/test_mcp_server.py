from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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
        mock_store.search.assert_called_once_with("renewables", top_k=5)
        
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

# TTL in vector store server 

def test_vector_store_add_triggers_with_documents():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        client = get_vs_client()
        docs = [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        r = client.post("/vector_store/add", json={"documents": docs})
        assert r.status_code == 200
        mock_store.add_documents.assert_called_once()


def test_vector_store_search_returns_scored_results():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = [
            {"content": "Solar grew 40%.", "score": 0.85}
        ]
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar", "top_k": 3})
        results = r.json()["result"]
        assert len(results) > 0
        assert "score" in results[0]


def test_vector_store_search_passes_top_k():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        client = get_vs_client()
        client.post("/vector_store/search", json={"query": "solar", "top_k": 3})
        mock_store.search.assert_called_once_with("solar", top_k=3)


def test_vector_store_search_default_top_k():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        client = get_vs_client()
        client.post("/vector_store/search", json={"query": "solar"})
        mock_store.search.assert_called_once_with("solar", top_k=5)

# reranked flag in vector store server 

def test_vector_store_search_returns_reranked_flag():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = [
            {"content": "Solar grew 40%.", "score": 0.85}
        ]
        mock_store.reranker = MagicMock()
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar"})
        assert "reranked" in r.json()


def test_vector_store_search_reranked_true_when_reranker_available():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = MagicMock()
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar"})
        assert r.json()["reranked"] is True


def test_vector_store_search_reranked_false_when_reranker_unavailable():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = None
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar"})
        assert r.json()["reranked"] is False


def test_vector_store_search_accepts_rerank_flag():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = None
        client = get_vs_client()
        r = client.post("/vector_store/search", json={"query": "solar", "rerank": False})
        assert r.status_code == 200

# ── vector store server startup warmup ───────────────────────────────────────

def test_vector_store_server_prewarms_reranker_on_startup():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = MagicMock()
        from fastapi.testclient import TestClient
        from src.mcp.servers.vector_store_server import app
        with TestClient(app) as client:
            mock_store.search.assert_called_with("warmup", top_k=1)


def test_vector_store_server_startup_handles_empty_store():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = None
        from fastapi.testclient import TestClient
        from src.mcp.servers.vector_store_server import app
        try:
            with TestClient(app) as client:
                pass
        except Exception:
            assert False, "startup should not raise when store is empty"

# --- unknown server and tool ---

def test_unknown_server_returns_empty():
    from src.mcp.client.mcp_client import MCPClient
    client = MCPClient()
    result = client.call_tool("nonexistent_server", "search", {"query": "test"})
    assert result == []

def test_unknown_tool_returns_empty():
    from src.mcp.client.mcp_client import MCPClient
    client = MCPClient()
    result = client.call_tool("vector_store", "nonexistent_tool", {"query": "test"})
    assert result == []

def test_unknown_server_logs_warning(capsys):
    from src.mcp.client.mcp_client import MCPClient
    client = MCPClient()
    client.call_tool("bad_server", "search", {})
    assert "unknown server" in capsys.readouterr().out

def test_unknown_tool_logs_warning(capsys):
    from src.mcp.client.mcp_client import MCPClient
    client = MCPClient()
    client.call_tool("vector_store", "bad_tool", {})
    assert "unknown tool" in capsys.readouterr().out


# --- tool name mapping ---

def test_vector_store_search_maps_to_mcp_search():
    from src.mcp.client.mcp_client import TOOL_NAMES
    assert TOOL_NAMES[("vector_store", "search")] == "mcp_search"

def test_vector_store_add_maps_to_mcp_add():
    from src.mcp.client.mcp_client import TOOL_NAMES
    assert TOOL_NAMES[("vector_store", "add")] == "mcp_add"

def test_web_search_maps_to_mcp_web_search():
    from src.mcp.client.mcp_client import TOOL_NAMES
    assert TOOL_NAMES[("web_search", "search")] == "mcp_web_search"

def test_arxiv_maps_to_mcp_arxiv_search():
    from src.mcp.client.mcp_client import TOOL_NAMES
    assert TOOL_NAMES[("arxiv", "search")] == "mcp_arxiv_search"


# --- server URLs ---

def test_vector_store_url_contains_mcp():
    from src.mcp.client.mcp_client import SERVER_URLS
    assert "/mcp" in SERVER_URLS["vector_store"]
    assert "8011" in SERVER_URLS["vector_store"]

def test_web_search_url_contains_mcp():
    from src.mcp.client.mcp_client import SERVER_URLS
    assert "/mcp" in SERVER_URLS["web_search"]
    assert "8012" in SERVER_URLS["web_search"]

def test_arxiv_url_contains_mcp():
    from src.mcp.client.mcp_client import SERVER_URLS
    assert "/mcp" in SERVER_URLS["arxiv"]
    assert "8013" in SERVER_URLS["arxiv"]
            
# --- result parsing ---

def _make_mock_session(text: str):
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.content = [block]
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=result)
    return session

def _patch_http(session):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock(), AsyncMock()))
    cm.__aexit__ = AsyncMock(return_value=False)
    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return cm, session_cm

def test_call_tool_parses_list_result():
    from src.mcp.client.mcp_client import MCPClient
    session = _make_mock_session("[{'content': 'solar data', 'score': 0.9}]")
    sse_cm, session_cm = _patch_http(session)
    with patch("src.mcp.client.mcp_client.streamablehttp_client", return_value=sse_cm):
        with patch("src.mcp.client.mcp_client.ClientSession", return_value=session_cm):
            client = MCPClient()
            result = client.call_tool("vector_store", "search", {"query": "solar"})
    assert isinstance(result, list)
    assert result[0]["content"] == "solar data"

def test_call_tool_returns_empty_list_on_empty_result():
    from src.mcp.client.mcp_client import MCPClient
    session = _make_mock_session("[]")
    sse_cm, session_cm = _patch_http(session)
    with patch("src.mcp.client.mcp_client.streamablehttp_client", return_value=sse_cm):
        with patch("src.mcp.client.mcp_client.ClientSession", return_value=session_cm):
            client = MCPClient()
            result = client.call_tool("vector_store", "search", {"query": "obscure"})
    assert result == []
    
def test_call_tool_returns_empty_on_connection_error():
    from src.mcp.client.mcp_client import MCPClient
    with patch("src.mcp.client.mcp_client.asyncio.run", side_effect=Exception("connection refused")):
        client = MCPClient()
        result = client.call_tool("vector_store", "search", {"query": "test"})
    assert result == []

def test_call_tool_logs_on_connection_error(capsys):
    from src.mcp.client.mcp_client import MCPClient
    with patch("src.mcp.client.mcp_client.asyncio.run", side_effect=Exception("connection refused")):
        client = MCPClient()
        client.call_tool("vector_store", "search", {"query": "test"})
    assert "call failed" in capsys.readouterr().out