from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def get_arxiv_client():
    from src.mcp.servers.arxiv_server import app
    return TestClient(app)


FAKE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Solar Energy Advances in 2024</title>
    <summary>This paper explores recent advances in solar energy technology including bifacial panels and perovskite cells.</summary>
    <id>http://arxiv.org/abs/2401.00001</id>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
  <entry>
    <title>Wind Energy Storage Solutions</title>
    <summary>A comprehensive review of wind energy storage solutions including flow batteries and pumped hydro.</summary>
    <id>http://arxiv.org/abs/2401.00002</id>
    <published>2024-01-14T00:00:00Z</published>
  </entry>
</feed>"""

EMPTY_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


# endpoint 

def test_arxiv_search_returns_200():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        assert r.status_code == 200


def test_arxiv_search_returns_result_key():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        assert "result" in r.json()


def test_arxiv_search_missing_topic_returns_422():
    client = get_arxiv_client()
    r = client.post("/arxiv/search", json={})
    assert r.status_code == 422


def test_arxiv_search_returns_correct_number_of_results():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy", "max_results": 5})
        results = r.json()["result"]
        assert len(results) == 2


def test_arxiv_search_returns_empty_on_failure():
    with patch("src.mcp.servers.arxiv_server.httpx.get", side_effect=Exception("timeout")):
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        assert r.status_code == 200
        assert r.json()["result"] == []


def test_arxiv_search_default_max_results():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        client.post("/arxiv/search", json={"topic": "solar energy"})
        url_called = mock_get.call_args[0][0]
        assert "max_results=5" in url_called

# result structure 

def test_arxiv_results_have_required_keys():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        results = r.json()["result"]
        assert all(k in doc for doc in results for k in ["title", "content", "source", "published"])


def test_arxiv_results_have_correct_title():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        titles = [doc["title"] for doc in r.json()["result"]]
        assert "Solar Energy Advances in 2024" in titles


def test_arxiv_results_have_correct_source():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        sources = [doc["source"] for doc in r.json()["result"]]
        assert "http://arxiv.org/abs/2401.00001" in sources


def test_arxiv_results_content_truncated_to_500():
    long_summary = "A " * 400
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Long Paper</title>
    <summary>{long_summary}</summary>
    <id>http://arxiv.org/abs/2401.00003</id>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
</feed>"""
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=xml)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        content = r.json()["result"][0]["content"]
        assert len(content) <= 500


def test_arxiv_results_published_date_truncated():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "solar energy"})
        for doc in r.json()["result"]:
            assert len(doc["published"]) == 10


def test_arxiv_empty_feed_returns_empty_list():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=EMPTY_ARXIV_XML)
        client = get_arxiv_client()
        r = client.post("/arxiv/search", json={"topic": "obscure topic"})
        assert r.json()["result"] == []


# API call structure 

def test_arxiv_calls_correct_url():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=EMPTY_ARXIV_XML)
        client = get_arxiv_client()
        client.post("/arxiv/search", json={"topic": "solar energy"})
        url_called = mock_get.call_args[0][0]
        assert "arxiv.org" in url_called


def test_arxiv_sorts_by_submitted_date():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=EMPTY_ARXIV_XML)
        client = get_arxiv_client()
        client.post("/arxiv/search", json={"topic": "solar energy"})
        url_called = mock_get.call_args[0][0]
        assert "sortBy=submittedDate" in url_called
        assert "sortOrder=descending" in url_called

def test_arxiv_search_query_includes_topic():
    with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=EMPTY_ARXIV_XML)
        client = get_arxiv_client()
        client.post("/arxiv/search", json={"topic": "solar energy"})
        url_called = mock_get.call_args[0][0]
        assert "solar" in url_called
        assert "energy" in url_called

# MCP client routing 

def test_mcp_client_routes_arxiv_to_8013():
    from src.mcp.client.mcp_client import SERVER_URLS
    assert "8013" in SERVER_URLS["arxiv"]

def test_mcp_client_arxiv_uses_correct_url():
    from src.mcp.client.mcp_client import SERVER_URLS
    assert "8013" in SERVER_URLS["arxiv"]
    assert "/mcp" in SERVER_URLS["arxiv"]
            
# researcher integration

def test_researcher_merges_web_and_arxiv_results():
    web_docs = [{"title": "Web Solar", "content": "Web content.", "source": "http://web.com"}]
    arxiv_docs = [{"title": "arXiv Solar", "content": "Paper content.", "source": "http://arxiv.org/abs/1"}]

    call_map = {
        "web_search": web_docs,
        "arxiv": arxiv_docs,
        "vector_store": []
    }

    def fake_call_tool(server, tool, arguments):
        return call_map.get(server, [])

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", side_effect=fake_call_tool):
        from src.agents.researcher import ResearchAgent
        docs = ResearchAgent().search("solar energy")
        titles = [d["title"] for d in docs]
        assert "Web Solar" in titles
        assert "arXiv Solar" in titles


def test_researcher_continues_if_arxiv_unavailable():
    web_docs = [{"title": "Web Solar", "content": "Web content.", "source": "http://web.com"}]

    def fake_call_tool(server, tool, arguments):
        if server == "arxiv":
            raise Exception("arxiv unavailable")
        if server == "web_search":
            return web_docs
        return []

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", side_effect=fake_call_tool):
        from src.agents.researcher import ResearchAgent
        docs = ResearchAgent().search("solar energy")
        assert len(docs) > 0
        assert docs[0]["title"] == "Web Solar"


# docker 

def test_compose_has_arxiv_service():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    assert "mcp_arxiv" in compose.get("services", {})


def test_compose_arxiv_correct_port():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    arxiv = compose["services"].get("mcp_arxiv", {})
    assert any("8003" in str(p) for p in arxiv.get("ports", []))


def test_compose_arxiv_depends_on_ollama():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    arxiv = compose["services"].get("mcp_arxiv", {})
    assert "ollama" in arxiv.get("depends_on", [])