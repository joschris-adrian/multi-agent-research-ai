import os
import importlib
from unittest.mock import patch, MagicMock
from src.agents.planner import PlannerAgent
from src.agents.writer import WriterAgent
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearchAgent


def make_mock(text="some response"):
    return MagicMock(json=lambda: {"response": text})


@patch("src.agents.base_agent.requests.post")
def test_planner_returns_string(mock_post):
    mock_post.return_value = make_mock("1. Search\n2. Analyse\n3. Write")
    result = PlannerAgent().plan("What are AI trends?")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_writer_returns_string(mock_post):
    mock_post.return_value = make_mock("# Report\nAI is growing.")
    result = WriterAgent().write_report("AI is growing rapidly")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_critic_returns_string(mock_post):
    mock_post.return_value = make_mock("Looks good, could add more sources.")
    result = CriticAgent().review("Sample report")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_analyst_returns_string(mock_post):
    mock_post.return_value = make_mock("Solar growing 20% YoY")
    docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
    result = AnalystAgent().analyze(docs, "renewable energy")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_researcher_extract_query(mock_post):
    mock_post.return_value = make_mock("latest renewable energy trends 2025")
    result = ResearchAgent().extract_query("1. Find trends\n2. Analyse", "energy trends?")
    assert isinstance(result, str) and len(result) > 0


@patch("ddgs.DDGS.text")
def test_researcher_search_structure(mock_ddgs):
    mock_ddgs.return_value = [
        {"title": "Solar Boom", "body": "Solar is growing fast.", "href": "http://example.com"}
    ]
    docs = ResearchAgent().search("solar energy trends")
    assert isinstance(docs, list)
    assert all(k in docs[0] for k in ["title", "content", "source"])


@patch("src.agents.base_agent.requests.post")
def test_mocked_base_agent(mock_post):
    mock_post.return_value = make_mock("Mocked response")
    result = PlannerAgent().plan("Test")
    assert result == "Mocked response"


@patch("src.agents.base_agent.requests.post")
def test_ollama_host_env_var(mock_post):
    mock_post.return_value = make_mock("ok")
    with patch.dict(os.environ, {"OLLAMA_HOST": "http://ollama:11434"}):
        import src.agents.base_agent as base_module
        importlib.reload(base_module)
        agent = base_module.BaseAgent(role="Test", goal="Test")
        agent.run("test")
        assert "ollama:11434" in mock_post.call_args[0][0]
    importlib.reload(base_module)


# retry logic
@patch("ddgs.DDGS.text")
def test_search_retries_on_failure(mock_ddgs):
    # fail twice, succeed on third attempt
    mock_ddgs.side_effect = [
        Exception("rate limited"),
        Exception("rate limited"),
        [{"title": "Solar", "body": "Solar is growing.", "href": "http://example.com"}]
    ]
    docs = ResearchAgent().search("solar energy", retries=3, delay=0)
    assert len(docs) == 1


@patch("ddgs.DDGS.text")
def test_search_returns_empty_after_all_retries_fail(mock_ddgs):
    mock_ddgs.side_effect = Exception("rate limited")
    docs = ResearchAgent().search("solar energy", retries=3, delay=0)
    assert docs == []
