from unittest.mock import patch
import pytest

with patch("src.memory.vector_store.VectorStore") as mock_vs:
    mock_vs.return_value.search.return_value = []
    mock_vs.return_value.reranker = None
    from src.mcp.servers.vector_store_server import mcp, store


# --- tool registration ---

def test_mcp_search_tool_is_registered():
    names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "mcp_search" in names

def test_mcp_add_tool_is_registered():
    names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "mcp_add" in names

def test_mcp_has_exactly_two_tools():
    assert len(mcp._tool_manager._tools) == 2

def test_mcp_search_tool_has_description():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
    assert tool.description is not None
    assert len(tool.description) > 0

def test_mcp_add_tool_has_description():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
    assert tool.description is not None
    assert len(tool.description) > 0

def test_mcp_search_schema_has_query_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
    schema = tool.parameters
    assert "query" in schema.get("properties", {})

def test_mcp_search_schema_has_top_k_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
    schema = tool.parameters
    assert "top_k" in schema.get("properties", {})

def test_mcp_search_schema_query_is_required():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
    schema = tool.parameters
    assert "query" in schema.get("required", [])

def test_mcp_add_schema_has_documents_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
    schema = tool.parameters
    assert "documents" in schema.get("properties", {})


# --- tool execution ---

def test_mcp_search_calls_store():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
        tool.fn(query="solar energy", top_k=5)
    mock_store.search.assert_called_once_with("solar energy", top_k=5)

def test_mcp_search_respects_top_k():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
        tool.fn(query="solar energy", top_k=10)
    mock_store.search.assert_called_once_with("solar energy", top_k=10)

def test_mcp_search_returns_string():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = [{"content": "solar data", "score": 0.9}]
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
        result = tool.fn(query="solar energy", top_k=5)
    assert isinstance(result, str)
    assert "solar data" in result

def test_mcp_search_returns_empty_string_on_no_results():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
        result = tool.fn(query="obscure topic", top_k=5)
    assert isinstance(result, str)

def test_mcp_search_raises_on_store_exception():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.side_effect = Exception("ChromaDB unavailable")
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
        with pytest.raises(ValueError, match="ChromaDB unavailable"):
            tool.fn(query="test", top_k=5)

def test_mcp_add_calls_store():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.add_documents.return_value = None
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
        tool.fn(documents=[{"content": "test", "source": "http://example.com", "title": "Test"}])
    mock_store.add_documents.assert_called_once()

def test_mcp_add_returns_ok():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.add_documents.return_value = None
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
        result = tool.fn(documents=[{"content": "test", "source": "http://example.com", "title": "Test"}])
    assert result == "ok"

def test_mcp_add_raises_on_store_exception():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.add_documents.side_effect = Exception("write failed")
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
        with pytest.raises(ValueError, match="write failed"):
            tool.fn(documents=[{"content": "test"}])


# --- existing endpoints unaffected ---

from fastapi.testclient import TestClient
with patch("src.memory.vector_store.VectorStore") as mock_vs2:
    mock_vs2.return_value.search.return_value = []
    mock_vs2.return_value.reranker = None
    from src.mcp.servers.vector_store_server import app
client = TestClient(app)

def test_existing_search_endpoint_still_works():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.search.return_value = []
        mock_store.reranker = None
        response = client.post("/vector_store/search", json={"query": "test"})
    assert response.status_code == 200

def test_existing_add_endpoint_still_works():
    with patch("src.mcp.servers.vector_store_server.store") as mock_store:
        mock_store.add_documents.return_value = None
        response = client.post("/vector_store/add", json={"documents": [{"content": "test", "source": "http://example.com", "title": "Test"}]})
    assert response.status_code == 200

def test_mcp_mount_exists():
    paths = [r.path for r in app.routes]
    assert "/mcp" in paths