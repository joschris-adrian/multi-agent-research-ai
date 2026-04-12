from unittest.mock import patch, MagicMock
from src.workflow.agent_pipeline import MultiAgentResearchSystem


def mock_response(text="mocked agent response"):
    return MagicMock(json=lambda: {"response": text})


@patch("src.agents.base_agent.requests.post")
def test_pipeline_returns_expected_keys(mock_post):
    mock_post.return_value = mock_response()
    result = MultiAgentResearchSystem().run("renewable energy trends")
    for key in ["question", "tasks", "documents", "insights", "report", "critic_feedback"]:
        assert key in result


@patch("src.agents.base_agent.requests.post")
def test_pipeline_preserves_question(mock_post):
    mock_post.return_value = mock_response()
    q = "What are the latest trends in renewable energy?"
    result = MultiAgentResearchSystem().run(q)
    assert result["question"] == q


@patch("src.agents.base_agent.requests.post")
def test_pipeline_empty_question(mock_post):
    mock_post.return_value = mock_response()
    result = MultiAgentResearchSystem().run("")
    assert result is not None


@patch("src.agents.base_agent.requests.post")
def test_pipeline_short_question(mock_post):
    mock_post.return_value = mock_response()
    result = MultiAgentResearchSystem().run("AI")
    assert "report" in result


@patch("src.agents.base_agent.requests.post")
def test_all_agents_called(mock_post):
    mock_post.return_value = mock_response()
    system = MultiAgentResearchSystem()

    with patch.object(system.planner, "plan", return_value="tasks") as p1, \
         patch.object(system.researcher, "extract_query", return_value="query") as p2, \
         patch.object(system.researcher, "search", return_value=[]) as p3, \
         patch.object(system.analyst, "analyze", return_value="insights") as p4, \
         patch.object(system.writer, "write_report", return_value="report") as p5, \
         patch.object(system.critic, "review", return_value="feedback") as p6:

        system.run("test")

        for agent_call in [p1, p2, p3, p4, p5, p6]:
            agent_call.assert_called_once()
