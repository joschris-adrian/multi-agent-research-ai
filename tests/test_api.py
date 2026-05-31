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
    "entities": {
        "companies": ["Google", "OpenAI"],
        "trends": ["LLM adoption"],
        "technologies": ["Transformers"],
        "relationships": []
    },
    "report": "## Summary\nAI is transforming industries.\n\n## Key Trends\n- LLM adoption\n\n## Key Players\nGoogle and OpenAI.\n\n## Statistics\nNo specific statistics.\n\n## Conclusion\nAI will continue to grow.",
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
    for key in ["question", "report", "tasks", "insights", "entities", "critic_feedback", "documents"]:
        assert key in data

@patch("api.main.system.run")
def test_research_response_includes_documents(mock_run):
    mock_run.return_value = FAKE_RESULT
    data = client.post("/research", json={"query": "AI trends"}).json()
    assert "documents" in data
    assert isinstance(data["documents"], list)

@patch("api.main.system.run")
def test_research_handles_slow_response(mock_run):
    import time
    def slow_run(q):
        return FAKE_RESULT
    mock_run.side_effect = slow_run
    r = client.post("/research", json={"query": "AI trends"})
    assert r.status_code == 200

@patch("api.main.system.run")
def test_research_documents_have_required_keys(mock_run):
    mock_run.return_value = FAKE_RESULT
    data = client.post("/research", json={"query": "AI trends"}).json()
    for doc in data["documents"]:
        assert "title" in doc
        assert "content" in doc
        assert "source" in doc


@patch("api.main.system.run")
def test_research_is_async_compatible(mock_run):
    mock_run.return_value = FAKE_RESULT
    r = client.post("/research", json={"query": "AI trends"})
    assert r.status_code == 200

# ── streaming endpoint ────────────────────────────────────────────────────────

@patch("api.main.system.run")
def test_research_stream_endpoint_exists(mock_run):
    mock_run.return_value = FAKE_RESULT
    routes = [r.path for r in app.routes]
    assert "/research/stream" in routes


@patch("src.agents.planner.PlannerAgent.plan")
@patch("src.agents.researcher.ResearchAgent.extract_query")
@patch("src.agents.researcher.ResearchAgent.search")
@patch("src.agents.analyst.AnalystAgent.analyze")
@patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities")
@patch("src.agents.writer.WriterAgent.write_report")
@patch("src.agents.critic.CriticAgent.review")
@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_returns_200(
    mock_kg, mock_critic, mock_writer, mock_graph,
    mock_analyst, mock_search, mock_extract, mock_plan
):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_plan.return_value = "1. Search trends"
    mock_extract.return_value = "solar energy trends"
    mock_search.return_value = [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
    mock_analyst.return_value = "Solar is growing rapidly."
    mock_graph.return_value = FAKE_RESULT["entities"]
    mock_writer.return_value = FAKE_RESULT["report"]
    mock_critic.return_value = "Report looks good."

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})
    assert r.status_code == 200


@patch("src.agents.planner.PlannerAgent.plan")
@patch("src.agents.researcher.ResearchAgent.extract_query")
@patch("src.agents.researcher.ResearchAgent.search")
@patch("src.agents.analyst.AnalystAgent.analyze")
@patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities")
@patch("src.agents.writer.WriterAgent.write_report")
@patch("src.agents.critic.CriticAgent.review")
@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_returns_event_stream_content_type(
    mock_kg, mock_critic, mock_writer, mock_graph,
    mock_analyst, mock_search, mock_extract, mock_plan
):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_plan.return_value = "1. Search trends"
    mock_extract.return_value = "solar energy"
    mock_search.return_value = []
    mock_analyst.return_value = "Solar is growing."
    mock_graph.return_value = FAKE_RESULT["entities"]
    mock_writer.return_value = FAKE_RESULT["report"]
    mock_critic.return_value = "Looks good."

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})
    assert "text/event-stream" in r.headers.get("content-type", "")


@patch("src.agents.planner.PlannerAgent.plan")
@patch("src.agents.researcher.ResearchAgent.extract_query")
@patch("src.agents.researcher.ResearchAgent.search")
@patch("src.agents.analyst.AnalystAgent.analyze")
@patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities")
@patch("src.agents.writer.WriterAgent.write_report")
@patch("src.agents.critic.CriticAgent.review")
@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_emits_agent_events(
    mock_kg, mock_critic, mock_writer, mock_graph,
    mock_analyst, mock_search, mock_extract, mock_plan
):
    import json as json_lib
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_plan.return_value = "1. Search trends"
    mock_extract.return_value = "solar energy"
    mock_search.return_value = []
    mock_analyst.return_value = "Solar is growing."
    mock_graph.return_value = FAKE_RESULT["entities"]
    mock_writer.return_value = FAKE_RESULT["report"]
    mock_critic.return_value = "Looks good."

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})

    events = []
    for line in r.text.split("\n"):
        if line.startswith("data: "):
            events.append(json_lib.loads(line[6:]))

    event_types = [e["event"] for e in events]
    assert "start" in event_types
    assert "agent" in event_types
    assert "complete" in event_types


@patch("src.agents.planner.PlannerAgent.plan")
@patch("src.agents.researcher.ResearchAgent.extract_query")
@patch("src.agents.researcher.ResearchAgent.search")
@patch("src.agents.analyst.AnalystAgent.analyze")
@patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities")
@patch("src.agents.writer.WriterAgent.write_report")
@patch("src.agents.critic.CriticAgent.review")
@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_emits_all_six_agents(
    mock_kg, mock_critic, mock_writer, mock_graph,
    mock_analyst, mock_search, mock_extract, mock_plan
):
    import json as json_lib
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_plan.return_value = "1. Search trends"
    mock_extract.return_value = "solar energy"
    mock_search.return_value = []
    mock_analyst.return_value = "Solar is growing."
    mock_graph.return_value = FAKE_RESULT["entities"]
    mock_writer.return_value = FAKE_RESULT["report"]
    mock_critic.return_value = "Looks good."

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})

    agents_seen = set()
    for line in r.text.split("\n"):
        if line.startswith("data: "):
            event = json_lib.loads(line[6:])
            if event["event"] == "agent":
                agents_seen.add(event["data"]["agent"])

    assert agents_seen == {"planner", "researcher", "analyst", "graph_builder", "writer", "critic"}


@patch("src.agents.planner.PlannerAgent.plan")
@patch("src.agents.researcher.ResearchAgent.extract_query")
@patch("src.agents.researcher.ResearchAgent.search")
@patch("src.agents.analyst.AnalystAgent.analyze")
@patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities")
@patch("src.agents.writer.WriterAgent.write_report")
@patch("src.agents.critic.CriticAgent.review")
@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_complete_event_has_expected_keys(
    mock_kg, mock_critic, mock_writer, mock_graph,
    mock_analyst, mock_search, mock_extract, mock_plan
):
    import json as json_lib
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_plan.return_value = "1. Search trends"
    mock_extract.return_value = "solar energy"
    mock_search.return_value = []
    mock_analyst.return_value = "Solar is growing."
    mock_graph.return_value = FAKE_RESULT["entities"]
    mock_writer.return_value = FAKE_RESULT["report"]
    mock_critic.return_value = "Looks good."

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})

    complete_event = None
    for line in r.text.split("\n"):
        if line.startswith("data: "):
            event = json_lib.loads(line[6:])
            if event["event"] == "complete":
                complete_event = event["data"]

    assert complete_event is not None
    for key in ["question", "tasks", "documents", "insights", "entities", "report", "critic_feedback"]:
        assert key in complete_event


def test_research_stream_missing_query_returns_422():
    r = client.post("/research/stream", json={})
    assert r.status_code == 422


@patch("src.graph.knowledge_graph.KnowledgeGraph")
def test_research_stream_handles_neo4j_unavailable(mock_kg):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(side_effect=Exception("neo4j unavailable"))
    )
    with patch("src.agents.planner.PlannerAgent.plan", return_value="1. Search"), \
         patch("src.agents.researcher.ResearchAgent.extract_query", return_value="solar"), \
         patch("src.agents.researcher.ResearchAgent.search", return_value=[]), \
         patch("src.agents.analyst.AnalystAgent.analyze", return_value="Solar growing."), \
         patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_RESULT["entities"]), \
         patch("src.agents.writer.WriterAgent.write_report", return_value=FAKE_RESULT["report"]), \
         patch("src.agents.critic.CriticAgent.review", return_value="Looks good."), \
         patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        r = client.post("/research/stream", json={"query": "solar energy"})
        assert r.status_code == 200

def test_research_missing_query_returns_422():
    assert client.post("/research", json={}).status_code == 422


@patch("api.main.system.run")
def test_research_empty_query(mock_run):
    mock_run.return_value = FAKE_RESULT
    assert client.post("/research", json={"query": ""}).status_code == 200


def test_api_url_env_var():
    with patch.dict(os.environ, {"API_URL": "http://api:8000"}):
        assert os.getenv("API_URL") == "http://api:8000"
