import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# helpers 

def make_mock(text="some response"):
    return MagicMock(json=lambda: {"response": text})


FAKE_ENTITIES = {
    "companies": ["Tesla", "SolarEdge"],
    "trends": ["Solar growth"],
    "technologies": ["Battery storage"],
    "relationships": []
}

FAKE_REPORT = """## Summary
Solar energy is growing rapidly.

## Key Trends
- 585 GW added last year

## Key Players
No specific organisations identified.

## Statistics
- 585 GW added last year

## Conclusion
Solar will continue to grow."""


# agent server endpoints 

def get_a2a_client():
    with patch("src.agents.base_agent.requests.post") as mock_post:
        mock_post.return_value = make_mock("mocked response")
        with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
            from src.a2a.agent_server import app
            return TestClient(app)


@patch("src.agents.base_agent.requests.post")
def test_a2a_planner_endpoint_returns_200(mock_post):
    mock_post.return_value = make_mock("1. Search trends\n2. Analyse data")
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        from src.a2a.agent_server import app
        client = TestClient(app)
        r = client.post("/agent/planner", json={"question": "What are AI trends?"})
        assert r.status_code == 200


@patch("src.agents.base_agent.requests.post")
def test_a2a_planner_endpoint_returns_result_key(mock_post):
    mock_post.return_value = make_mock("1. Search trends")
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        from src.a2a.agent_server import app
        client = TestClient(app)
        r = client.post("/agent/planner", json={"question": "What are AI trends?"})
        assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_planner_missing_question_returns_422(mock_post):
    mock_post.return_value = make_mock("1. Search trends")
    from src.a2a.agent_server import app
    client = TestClient(app)
    r = client.post("/agent/planner", json={})
    assert r.status_code == 422


@patch("src.agents.base_agent.requests.post")
def test_a2a_analyst_endpoint_returns_200(mock_post):
    mock_post.return_value = make_mock("Solar growing 20% YoY")
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        from src.a2a.agent_server import app
        client = TestClient(app)
        r = client.post("/agent/analyst", json={
            "documents": [{"title": "Solar", "content": "Solar booming.", "source": "http://example.com"}],
            "query": "solar energy"
        })
        assert r.status_code == 200
        assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_writer_endpoint_returns_200(mock_post):
    mock_post.return_value = make_mock(FAKE_REPORT)
    from src.a2a.agent_server import app
    client = TestClient(app)
    r = client.post("/agent/writer", json={
        "insights": "Solar is growing rapidly.",
        "entities": FAKE_ENTITIES
    })
    assert r.status_code == 200
    assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_critic_endpoint_returns_200(mock_post):
    mock_post.return_value = make_mock("Report looks good.")
    from src.a2a.agent_server import app
    client = TestClient(app)
    r = client.post("/agent/critic", json={"report": FAKE_REPORT})
    assert r.status_code == 200
    assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_graph_builder_endpoint_returns_200(mock_post):
    import json
    mock_post.return_value = make_mock(json.dumps(FAKE_ENTITIES))
    from src.a2a.agent_server import app
    client = TestClient(app)
    r = client.post("/agent/graph_builder", json={
        "insights": "Tesla uses battery storage.",
        "topic": "electric vehicles"
    })
    assert r.status_code == 200
    assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_researcher_endpoint_returns_200(mock_post):
    mock_post.return_value = make_mock("mocked")
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[
        {"title": "Solar", "content": "Growing.", "source": "http://example.com"}
    ]):
        from src.a2a.agent_server import app
        client = TestClient(app)
        r = client.post("/agent/researcher", json={"query": "solar energy"})
        assert r.status_code == 200
        assert "result" in r.json()


@patch("src.agents.base_agent.requests.post")
def test_a2a_all_endpoints_exist(mock_post):
    mock_post.return_value = make_mock("ok")
    from src.a2a.agent_server import app
    routes = [r.path for r in app.routes]
    assert "/agent/planner" in routes
    assert "/agent/researcher" in routes
    assert "/agent/analyst" in routes
    assert "/agent/writer" in routes
    assert "/agent/critic" in routes
    assert "/agent/graph_builder" in routes


# A2A client 

def test_a2a_client_calls_correct_endpoint():
    with patch("src.a2a.a2a_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"result": "tasks"})
        from src.a2a.a2a_client import A2AClient
        client = A2AClient()
        client.call_agent("planner", {"question": "AI trends?"})
        url = mock_post.call_args[0][0]
        assert "/agent/planner" in url


def test_a2a_client_uses_port_8004():
    with patch("src.a2a.a2a_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"result": "tasks"})
        from src.a2a.a2a_client import A2AClient
        client = A2AClient()
        client.call_agent("planner", {"question": "AI trends?"})
        url = mock_post.call_args[0][0]
        assert "8004" in url


def test_a2a_client_returns_result():
    with patch("src.a2a.a2a_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"result": "1. Search trends"})
        from src.a2a.a2a_client import A2AClient
        client = A2AClient()
        result = client.call_agent("planner", {"question": "AI trends?"})
        assert result == "1. Search trends"


def test_a2a_client_returns_none_on_failure():
    with patch("src.a2a.a2a_client.httpx.post", side_effect=Exception("connection refused")):
        from src.a2a.a2a_client import A2AClient
        client = A2AClient()
        result = client.call_agent("planner", {"question": "AI trends?"})
        assert result is None


def test_a2a_client_raises_on_unknown_agent():
    from src.a2a.a2a_client import A2AClient
    client = A2AClient()
    with pytest.raises(ValueError):
        client.call_agent("unknown_agent", {})


def test_a2a_client_all_agents_registered():
    from src.a2a.a2a_client import AGENT_ENDPOINTS
    for agent in ["planner", "researcher", "analyst", "writer", "critic", "graph_builder"]:
        assert agent in AGENT_ENDPOINTS


def test_a2a_client_handles_error_in_response():
    with patch("src.a2a.a2a_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"error": "agent failed", "result": ""}
        )
        from src.a2a.a2a_client import A2AClient
        client = A2AClient()
        result = client.call_agent("planner", {"question": "AI trends?"})
        assert result == ""


# A2A pipeline 

def make_a2a_mock(entities=None):
    entities = entities or FAKE_ENTITIES

    def fake_call_agent(agent_name, payload):
        if agent_name == "planner":
            return "1. Search trends\n2. Analyse data"
        if agent_name == "researcher":
            return [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        if agent_name == "analyst":
            return "Solar is growing rapidly with key statistics."
        if agent_name == "graph_builder":
            return entities
        if agent_name == "writer":
            return FAKE_REPORT
        if agent_name == "critic":
            return "Report is well structured."
        return None

    return fake_call_agent


@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_a2a_pipeline_returns_expected_keys(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    with patch("src.a2a.a2a_pipeline.A2AClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.call_agent.side_effect = make_a2a_mock()
        mock_client_class.return_value = mock_client

        from src.a2a.a2a_pipeline import A2AResearchSystem
        system = A2AResearchSystem()
        result = system.run("What are solar energy trends?")

        for key in ["question", "tasks", "documents", "insights", "entities", "report", "critic_feedback"]:
            assert key in result


@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_a2a_pipeline_preserves_question(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    with patch("src.a2a.a2a_pipeline.A2AClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.call_agent.side_effect = make_a2a_mock()
        mock_client_class.return_value = mock_client

        from src.a2a.a2a_pipeline import A2AResearchSystem
        system = A2AResearchSystem()
        result = system.run("What are solar energy trends?")
        assert result["question"] == "What are solar energy trends?"


@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_a2a_pipeline_rerequests_research_when_insights_insufficient(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    researcher_call_count = {"n": 0}
    analyst_call_count = {"n": 0}

    def fake_call_agent(agent_name, payload):
        if agent_name == "planner":
            return "1. Search trends"
        if agent_name == "researcher":
            researcher_call_count["n"] += 1
            return [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        if agent_name == "analyst":
            analyst_call_count["n"] += 1
            if analyst_call_count["n"] == 1:
                return "no information found on this topic"
            return "Solar is growing rapidly."
        if agent_name == "graph_builder":
            return FAKE_ENTITIES
        if agent_name == "writer":
            return FAKE_REPORT
        if agent_name == "critic":
            return "Report looks good."
        return None

    with patch("src.a2a.a2a_pipeline.A2AClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.call_agent.side_effect = fake_call_agent
        mock_client_class.return_value = mock_client

        from src.a2a.a2a_pipeline import A2AResearchSystem
        system = A2AResearchSystem()
        system.run("What are solar energy trends?")

        assert researcher_call_count["n"] == 2
        assert analyst_call_count["n"] == 2


@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_a2a_pipeline_revises_report_when_critic_flags_issues(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    writer_call_count = {"n": 0}
    critic_call_count = {"n": 0}

    def fake_call_agent(agent_name, payload):
        if agent_name == "planner":
            return "1. Search trends"
        if agent_name == "researcher":
            return [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        if agent_name == "analyst":
            return "Solar is growing rapidly."
        if agent_name == "graph_builder":
            return FAKE_ENTITIES
        if agent_name == "writer":
            writer_call_count["n"] += 1
            return FAKE_REPORT
        if agent_name == "critic":
            critic_call_count["n"] += 1
            if critic_call_count["n"] == 1:
                return "Report is missing key statistics and unclear in places."
            return "Report is now well structured."
        return None

    with patch("src.a2a.a2a_pipeline.A2AClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.call_agent.side_effect = fake_call_agent
        mock_client_class.return_value = mock_client

        from src.a2a.a2a_pipeline import A2AResearchSystem
        system = A2AResearchSystem()
        system.run("What are solar energy trends?")

        assert writer_call_count["n"] == 2
        assert critic_call_count["n"] == 2


@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_a2a_pipeline_handles_neo4j_unavailable(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(side_effect=Exception("neo4j unavailable"))
    )
    with patch("src.a2a.a2a_pipeline.A2AClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.call_agent.side_effect = make_a2a_mock()
        mock_client_class.return_value = mock_client

        from src.a2a.a2a_pipeline import A2AResearchSystem
        system = A2AResearchSystem()
        try:
            result = system.run("What are solar energy trends?")
            assert result is not None
        except Exception:
            assert False, "pipeline should not crash when neo4j is unavailable"


def test_a2a_pipeline_insufficient_insights_detection():
    from src.a2a.a2a_pipeline import A2AResearchSystem
    with patch("src.a2a.a2a_pipeline.KnowledgeGraph"):
        system = A2AResearchSystem()
        assert system._is_insufficient("no information found")
        assert system._is_insufficient("insufficient data available")
        assert system._is_insufficient("")
        assert system._is_insufficient(None)
        assert not system._is_insufficient("Solar is growing rapidly with 585 GW added.")


# docker 

def test_compose_has_a2a_service():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    assert "a2a" in compose.get("services", {})


def test_compose_a2a_correct_port():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    a2a = compose["services"].get("a2a", {})
    assert any("8004" in str(p) for p in a2a.get("ports", []))


def test_compose_a2a_depends_on_ollama():
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    a2a = compose["services"].get("a2a", {})
    assert "ollama" in a2a.get("depends_on", [])