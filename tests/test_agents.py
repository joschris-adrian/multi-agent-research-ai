import os
import importlib
from unittest.mock import patch, MagicMock
from src.agents.planner import PlannerAgent
from src.agents.writer import WriterAgent
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearchAgent
from src.agents.graph_builder import GraphBuilderAgent
from src.agents.base_agent import BaseAgent


def make_mock(text="some response"):
    return MagicMock(json=lambda: {"response": text})


FAKE_ENTITIES = {
    "companies": ["Tesla", "SolarEdge"],
    "trends": ["Solar growth"],
    "technologies": ["Battery storage"],
    "relationships": []
}


# ── BaseAgent defaults ────────────────────────────────────────────────────────

def test_base_agent_default_temperature():
    agent = BaseAgent(role="Test", goal="Test")
    assert agent.temperature == 0.7


def test_base_agent_default_max_tokens():
    agent = BaseAgent(role="Test", goal="Test")
    assert agent.max_tokens == 500


def test_base_agent_custom_temperature():
    agent = BaseAgent(role="Test", goal="Test", temperature=0.1)
    assert agent.temperature == 0.1


def test_base_agent_custom_max_tokens():
    agent = BaseAgent(role="Test", goal="Test", max_tokens=800)
    assert agent.max_tokens == 800


@patch("src.agents.base_agent.requests.post")
def test_base_agent_sends_system_separately(mock_post):
    mock_post.return_value = make_mock("ok")
    agent = BaseAgent(role="Tester", goal="Test")
    agent.run("hello")

    call_json = mock_post.call_args[1]["json"]
    assert "system" in call_json
    assert "prompt" in call_json
    assert call_json["prompt"] == "hello"
    assert "Role: Tester" in call_json["system"]


@patch("src.agents.base_agent.requests.post")
def test_base_agent_sends_options(mock_post):
    mock_post.return_value = make_mock("ok")
    agent = BaseAgent(role="Tester", goal="Test", temperature=0.3, max_tokens=200)
    agent.run("test prompt")

    options = mock_post.call_args[1]["json"]["options"]
    assert options["temperature"] == 0.3
    assert options["num_predict"] == 200
    assert "stop" in options


@patch("src.agents.base_agent.requests.post")
def test_base_agent_stream_is_false(mock_post):
    mock_post.return_value = make_mock("ok")
    agent = BaseAgent(role="Tester", goal="Test")
    agent.run("test")
    assert mock_post.call_args[1]["json"]["stream"] is False


# ── Per-agent temperature settings ───────────────────────────────────────────

def test_graph_builder_uses_low_temperature():
    assert GraphBuilderAgent().temperature == 0.1


def test_writer_uses_higher_temperature():
    assert WriterAgent().temperature == 0.8


def test_writer_has_larger_token_limit():
    assert WriterAgent().max_tokens == 800


# ── Writer uses entities ──────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_writer_includes_entities_in_prompt(mock_post):
    mock_post.return_value = make_mock("# Report\nTesla leads solar.")
    writer = WriterAgent()
    writer.write_report("Solar is growing.", FAKE_ENTITIES)

    prompt_sent = mock_post.call_args[1]["json"]["prompt"]
    assert "Tesla" in prompt_sent
    assert "SolarEdge" in prompt_sent
    assert "Solar growth" in prompt_sent
    assert "Battery storage" in prompt_sent


@patch("src.agents.base_agent.requests.post")
def test_writer_handles_empty_entities(mock_post):
    mock_post.return_value = make_mock("# Report\nSolar is growing.")
    writer = WriterAgent()
    result = writer.write_report("Solar is growing.", {})
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_writer_handles_no_entities_arg(mock_post):
    mock_post.return_value = make_mock("# Report\nSolar is growing.")
    writer = WriterAgent()
    result = writer.write_report("Solar is growing.")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_writer_falls_back_gracefully_when_entities_missing(mock_post):
    mock_post.return_value = make_mock("# Report")
    writer = WriterAgent()
    prompt_sent_before = None

    def capture(*args, **kwargs):
        nonlocal prompt_sent_before
        prompt_sent_before = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    writer.write_report("insights", {"companies": [], "trends": [], "technologies": []})
    assert "not identified" in prompt_sent_before


# ── Planner ───────────────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_planner_returns_string(mock_post):
    mock_post.return_value = make_mock("1. Search trends\n2. Analyse data")
    result = PlannerAgent().plan("What are AI trends?")
    assert isinstance(result, str) and len(result) > 0


# ── Critic ────────────────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_critic_returns_string(mock_post):
    mock_post.return_value = make_mock("Looks good.")
    result = CriticAgent().review("Sample report")
    assert isinstance(result, str) and len(result) > 0


# ── Analyst ───────────────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_analyst_returns_string(mock_post):
    mock_post.return_value = make_mock("Solar growing 20% YoY")
    docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
    result = AnalystAgent().analyze(docs, "renewable energy")
    assert isinstance(result, str) and len(result) > 0


# ── Researcher ────────────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_researcher_extract_query(mock_post):
    mock_post.return_value = make_mock("latest renewable energy trends 2025")
    result = ResearchAgent().extract_query("1. Find trends", "energy trends?")
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


# ── Retry logic ───────────────────────────────────────────────────────────────

@patch("ddgs.DDGS.text")
def test_search_retries_on_failure(mock_ddgs):
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


# ── Env variable: OLLAMA_HOST ─────────────────────────────────────────────────

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
