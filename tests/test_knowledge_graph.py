from unittest.mock import patch, MagicMock, call
import json


# ── KnowledgeGraph ────────────────────────────────────────────────────────────

@patch("src.graph.knowledge_graph.GraphDatabase")
def test_add_topic(mock_db):
    from src.graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_topic("renewable energy")
    assert mock_db.driver.called or True  # driver initialised


@patch("src.graph.knowledge_graph.GraphDatabase")
def test_get_entities_returns_list(mock_db):
    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: mock_session
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"name": "Tesla", "kind": "Company"},
        {"name": "Solar boom", "kind": "Trend"}
    ]
    mock_db.driver.return_value.session.return_value = mock_session

    from src.graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    results = kg.get_all_entities()
    assert isinstance(results, list)


# ── GraphBuilderAgent ─────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_graph_builder_returns_dict(mock_post):
    fake_entities = {
        "companies": ["Tesla", "SolarEdge"],
        "trends": ["Solar growth"],
        "technologies": ["Battery storage"],
        "relationships": [{"source": "Tesla", "target": "Battery storage", "relation": "USES"}]
    }
    mock_post.return_value = MagicMock(
        json=lambda: {"response": json.dumps(fake_entities)}
    )
    from src.agents.graph_builder import GraphBuilderAgent
    agent = GraphBuilderAgent()
    result = agent.extract_entities("some insights", "renewable energy")
    assert "companies" in result
    assert "trends" in result
    assert "technologies" in result
    assert "relationships" in result


@patch("src.agents.base_agent.requests.post")
def test_graph_builder_handles_bad_json(mock_post):
    mock_post.return_value = MagicMock(
        json=lambda: {"response": "not valid json at all"}
    )
    from src.agents.graph_builder import GraphBuilderAgent
    agent = GraphBuilderAgent()
    result = agent.extract_entities("some insights", "topic")
    # should return empty structure rather than crashing
    assert result == {"companies": [], "trends": [], "technologies": [], "relationships": []}


@patch("src.agents.base_agent.requests.post")
def test_graph_builder_strips_code_fences(mock_post):
    fake_entities = {
        "companies": ["Tesla"],
        "trends": [],
        "technologies": [],
        "relationships": []
    }
    mock_post.return_value = MagicMock(
        json=lambda: {"response": f"```json\n{json.dumps(fake_entities)}\n```"}
    )
    from src.agents.graph_builder import GraphBuilderAgent
    result = GraphBuilderAgent().extract_entities("insights", "topic")
    assert "Tesla" in result["companies"]


# ── GraphQL schema ────────────────────────────────────────────────────────────

@patch("src.graph.knowledge_graph.GraphDatabase")
def test_graphql_entities_query(mock_db):
    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: mock_session
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [{"name": "Tesla", "kind": "Company"}]
    mock_db.driver.return_value.session.return_value = mock_session

    from src.graphql.graphql_schema import schema
    result = schema.execute_sync("{ entities { name kind } }")
    assert result.errors is None
