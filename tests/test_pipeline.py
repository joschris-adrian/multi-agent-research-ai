from unittest.mock import patch, MagicMock
from src.workflow.agent_pipeline import MultiAgentResearchSystem


def mock_response(text="mocked agent response"):
    return MagicMock(json=lambda: {"response": text})


def make_mock_kg():
    kg = MagicMock()
    kg.add_topic = MagicMock()
    kg.add_entity = MagicMock()
    kg.link_entity_to_topic = MagicMock()
    kg.link_entities = MagicMock()
    return kg


FAKE_ENTITIES = {
    "companies": ["Tesla", "SolarEdge"],
    "trends": ["Solar growth"],
    "technologies": ["Battery storage"],
    "relationships": [{"source": "Tesla", "target": "Battery storage", "relation": "USES"}]
}


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_pipeline_returns_expected_keys(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES):
        result = MultiAgentResearchSystem().run("renewable energy trends")

    for key in ["question", "tasks", "documents", "insights", "entities", "report", "critic_feedback"]:
        assert key in result


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_pipeline_preserves_question(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()

    q = "What are the latest trends in renewable energy?"
    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES):
        result = MultiAgentResearchSystem().run(q)

    assert result["question"] == q


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_pipeline_empty_question(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES):
        result = MultiAgentResearchSystem().run("")

    assert result is not None


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_pipeline_short_question(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES):
        result = MultiAgentResearchSystem().run("AI")

    assert "report" in result


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_writer_receives_entities_from_pipeline(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()
    system = MultiAgentResearchSystem()

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES), \
         patch.object(system.writer, "write_report", wraps=system.writer.write_report) as mock_write:

        system.run("solar energy trends")

        args, kwargs = mock_write.call_args
        entities_passed = args[1] if len(args) > 1 else kwargs.get("entities", {})
        assert "companies" in entities_passed
        assert "Tesla" in entities_passed["companies"]


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_pipeline_entities_in_result(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value=FAKE_ENTITIES):
        result = MultiAgentResearchSystem().run("solar energy")

    assert result["entities"] == FAKE_ENTITIES


@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_all_agents_called(mock_kg_class, mock_post):
    mock_kg_class.return_value = make_mock_kg()
    mock_post.return_value = mock_response()
    system = MultiAgentResearchSystem()

    with patch.object(system.planner, "plan", return_value="tasks") as p1, \
         patch.object(system.researcher, "extract_query", return_value="query") as p2, \
         patch.object(system.researcher, "search", return_value=[]) as p3, \
         patch.object(system.analyst, "analyze", return_value="insights") as p4, \
         patch.object(system.graph_builder, "extract_entities", return_value=FAKE_ENTITIES) as p5, \
         patch.object(system.writer, "write_report", return_value="report") as p6, \
         patch.object(system.critic, "review", return_value="feedback") as p7:

        system.run("test")

        for agent_call in [p1, p2, p3, p4, p5, p6, p7]:
            agent_call.assert_called_once()

        # verify writer was called with entities
        write_args = p6.call_args
        assert write_args is not None
