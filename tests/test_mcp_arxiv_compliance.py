import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

MINIMAL_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper</title>
    <summary>This is a test abstract about solar energy research.</summary>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

with patch("httpx.get") as mock_get:
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = EMPTY_ATOM
    from src.mcp.servers.arxiv_server import mcp, app

client = TestClient(app)


# --- tool registration ---

def test_mcp_arxiv_search_tool_is_registered():
    names = [t.name for t in mcp._tool_manager._tools.values()]
    assert "mcp_arxiv_search" in names

def test_mcp_has_exactly_one_tool():
    assert len(mcp._tool_manager._tools) == 1

def test_mcp_arxiv_search_tool_has_description():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
    assert tool.description is not None
    assert len(tool.description) > 0

def test_mcp_arxiv_search_schema_has_topic_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
    assert "topic" in tool.parameters.get("properties", {})

def test_mcp_arxiv_search_schema_has_max_results_parameter():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
    assert "max_results" in tool.parameters.get("properties", {})

def test_mcp_arxiv_search_schema_topic_is_required():
    tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")
    assert "topic" in tool.parameters.get("required", [])


# --- tool execution ---

def _get_tool():
    return next(t for t in mcp._tool_manager._tools.values() if t.name == "mcp_arxiv_search")

def test_mcp_arxiv_search_returns_string():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MINIMAL_ATOM
        result = _get_tool().fn(topic="solar energy")
    assert isinstance(result, str)

def test_mcp_arxiv_search_returns_paper_title():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MINIMAL_ATOM
        result = _get_tool().fn(topic="solar energy")
    assert "Test Paper" in result

def test_mcp_arxiv_search_returns_empty_on_no_results():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = EMPTY_ATOM
        result = _get_tool().fn(topic="obscure topic")
    assert result == str([])

def test_mcp_arxiv_search_returns_empty_on_http_error():
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        result = _get_tool().fn(topic="solar energy")
    assert result == str([])

def test_mcp_arxiv_search_respects_max_results():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = EMPTY_ATOM
        _get_tool().fn(topic="solar energy", max_results=10)
    call_url = mock_get.call_args[0][0]
    assert "max_results=10" in call_url

def test_mcp_arxiv_search_uses_https():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = EMPTY_ATOM
        _get_tool().fn(topic="solar energy")
    call_url = mock_get.call_args[0][0]
    assert call_url.startswith("https://")

def test_mcp_arxiv_search_sorts_by_submitted_date():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = EMPTY_ATOM
        _get_tool().fn(topic="solar energy")
    call_url = mock_get.call_args[0][0]
    assert "sortBy=submittedDate" in call_url

def test_mcp_arxiv_search_truncates_content_to_500_chars():
    long_abstract = "x" * 1000
    atom = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Long Paper</title>
    <summary>{long_abstract}</summary>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>"""
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = atom
        result = _get_tool().fn(topic="test")
    parsed = eval(result)
    assert len(parsed[0]["content"]) <= 500


# --- existing endpoint unaffected ---

def test_existing_arxiv_endpoint_still_works():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = EMPTY_ATOM
        response = client.post("/arxiv/search", json={"topic": "solar energy"})
    assert response.status_code == 200

def test_existing_arxiv_endpoint_returns_result_key():
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MINIMAL_ATOM
        response = client.post("/arxiv/search", json={"topic": "solar energy"})
    assert "result" in response.json()

def test_mcp_tools_registered():
    assert len(mcp._tool_manager._tools) > 0