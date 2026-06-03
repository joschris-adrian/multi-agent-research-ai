import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

with patch("ddgs.DDGS") as mock_ddgs:
    mock_ddgs.return_value.__enter__.return_value.text.return_value = []
    from src.mcp.servers.web_search_server import mcp, app

client = TestClient(app)


# --- tool registration ---

def test_mcp_web_search_tool_is_registered():
    names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "mcp_web_search" in names

def test_mcp_has_exactly_one_tool():
    assert len(mcp._tool_manager._tools) == 1

def test_mcp_web_search_tool_has_description():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
    assert tool.description is not None
    assert len(tool.description) > 0

def test_mcp_web_search_schema_has_query_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
    assert "query" in tool.parameters.get("properties", {})

def test_mcp_web_search_schema_has_max_results_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
    assert "max_results" in tool.parameters.get("properties", {})

def test_mcp_web_search_schema_query_is_required():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")
    assert "query" in tool.parameters.get("required", [])


# --- tool execution ---

def _get_tool():
    return next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_web_search")

def _make_ddgs_result(title="Test", body="a" * 300, href="http://example.com"):
    return [{"title": title, "body": body, "href": href}]

def test_mcp_web_search_returns_string():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = _make_ddgs_result()
        result = _get_tool().fn(query="solar energy")
    assert isinstance(result, str)

def test_mcp_web_search_returns_chunks():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = _make_ddgs_result(body="a" * 300)
        result = _get_tool().fn(query="solar energy")
    assert "example.com" in result

def test_mcp_web_search_returns_empty_on_no_results():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        result = _get_tool().fn(query="solar energy", retries=1)
    assert result == str([])

def test_mcp_web_search_retries_on_failure():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        with patch("src.mcp.servers.web_search_server.time.sleep") as mock_sleep:
            mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("rate limited")
            result = _get_tool().fn(query="solar energy", retries=3, delay=1)
    assert mock_sleep.call_count == 2
    assert result == str([])

def test_mcp_web_search_respects_max_results():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = _make_ddgs_result()
        _get_tool().fn(query="solar energy", max_results=5)
    mock_instance.text.assert_called_once_with("solar energy", max_results=5)

def test_mcp_web_search_chunks_body_correctly():
    body = "x" * 500
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "T", "body": body, "href": "http://example.com"}
        ]
        result = _get_tool().fn(query="test", retries=1)
    parsed = eval(result)
    assert len(parsed) > 1
    for chunk in parsed:
        assert len(chunk["content"]) <= 200


# --- existing endpoint unaffected ---

def test_existing_search_endpoint_still_works():
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        response = client.post("/web_search/search", json={"query": "test", "retries": 1})
    assert response.status_code == 200

def test_mcp_tools_registered():
    assert len(mcp._tool_manager._tools) > 0