import pytest
from unittest.mock import patch, MagicMock


# --- vector store standalone server ---

def test_vector_store_mcp_server_registers_search_tool():
    with patch("src.memory.vector_store.VectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = []
        mock_vs.return_value.reranker = None
        from src.mcp.servers.vector_store_mcp_server import mcp
        names = [t.name for t in mcp._tool_manager._tools.values()]
        assert "mcp_search" in names

def test_vector_store_mcp_server_registers_add_tool():
    with patch("src.memory.vector_store.VectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = []
        mock_vs.return_value.reranker = None
        from src.mcp.servers.vector_store_mcp_server import mcp
        names = [t.name for t in mcp._tool_manager._tools.values()]
        assert "mcp_add" in names

def test_vector_store_mcp_server_search_calls_store():
    with patch("src.memory.vector_store.VectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = [{"content": "solar data", "score": 0.9}]
        mock_vs.return_value.reranker = None
        from src.mcp.servers.vector_store_mcp_server import mcp, store
        with patch("src.mcp.servers.vector_store_mcp_server.store") as mock_store:
            mock_store.search.return_value = [{"content": "solar data", "score": 0.9}]
            tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_search")
            result = tool.fn(query="solar energy", top_k=5)
        assert "solar data" in result

def test_vector_store_mcp_server_add_calls_store():
    with patch("src.memory.vector_store.VectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = []
        mock_vs.return_value.reranker = None
        from src.mcp.servers.vector_store_mcp_server import mcp
        with patch("src.mcp.servers.vector_store_mcp_server.store") as mock_store:
            mock_store.add_documents.return_value = None
            tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_add")
            result = tool.fn(documents=[{"content": "test", "source": "http://example.com", "title": "Test"}])
        assert result == "ok"

def test_vector_store_mcp_server_runs_on_port_8011():
    with patch("src.memory.vector_store.VectorStore") as mock_vs:
        mock_vs.return_value.search.return_value = []
        mock_vs.return_value.reranker = None
        import src.mcp.servers.vector_store_mcp_server as mod
        import inspect
        src = inspect.getsource(mod)
        assert "8011" in src


# --- web search standalone server ---

def test_web_search_mcp_server_registers_tool():
    from src.mcp.servers.web_search_mcp_server import mcp
    names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "mcp_web_search" in names

def test_web_search_mcp_server_search_returns_string():
    from src.mcp.servers.web_search_mcp_server import mcp
    with patch("src.mcp.servers.web_search_mcp_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Solar", "body": "Solar is growing fast.", "href": "http://example.com"}
        ]
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
        result = tool.fn(query="solar energy", retries=1)
    assert isinstance(result, str)

def test_web_search_mcp_server_returns_empty_on_failure():
    from src.mcp.servers.web_search_mcp_server import mcp
    with patch("src.mcp.servers.web_search_mcp_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("rate limited")
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
        result = tool.fn(query="solar energy", retries=1)
    assert result == str([])

def test_web_search_mcp_server_runs_on_port_8012():
    import src.mcp.servers.web_search_mcp_server as mod
    import inspect
    src = inspect.getsource(mod)
    assert "8012" in src


# --- arxiv standalone server ---

def test_arxiv_mcp_server_registers_tool():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        from src.mcp.servers.arxiv_mcp_server import mcp
        names = [t.name for t in mcp._tool_manager._tools.values()]
        assert "mcp_arxiv_search" in names

def test_arxiv_mcp_server_search_returns_string():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Solar Paper</title>
    <summary>Solar energy research.</summary>
    <id>http://arxiv.org/abs/2401.00001</id>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>"""
        from src.mcp.servers.arxiv_mcp_server import mcp
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
        result = tool.fn(topic="solar energy")
    assert "Solar Paper" in result

def test_arxiv_mcp_server_returns_empty_on_failure():
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        from src.mcp.servers.arxiv_mcp_server import mcp
        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
        result = tool.fn(topic="solar energy")
    assert result == str([])

def test_arxiv_mcp_server_runs_on_port_8013():
    import src.mcp.servers.arxiv_mcp_server as mod
    import inspect
    src = inspect.getsource(mod)
    assert "8013" in src


# --- port separation ---

def test_mcp_protocol_ports_are_separate_from_http_ports():
    from src.mcp.client.mcp_client import SERVER_URLS
    for url in SERVER_URLS.values():
        port = int(url.split(":")[2].split("/")[0])
        assert port >= 8011, f"MCP protocol port {port} overlaps with HTTP API ports"