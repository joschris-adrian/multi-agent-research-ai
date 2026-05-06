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


# BaseAgent defaults 

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


def test_researcher_search_structure():
    fake_docs = [
        {"title": "Solar Boom", "content": "Solar is growing fast.", "source": "http://example.com"}
    ]
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=fake_docs):
        docs = ResearchAgent().search("solar energy trends")
    assert isinstance(docs, list)
    assert all(k in docs[0] for k in ["title", "content", "source"])

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


# Per-agent temperature settings 

def test_graph_builder_uses_low_temperature():
    assert GraphBuilderAgent().temperature == 0.1


def test_writer_uses_higher_temperature():
    assert WriterAgent().temperature == 0.8


def test_writer_has_larger_token_limit():
    assert WriterAgent().max_tokens == 800


# Writer uses entities 

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


# Planner 

@patch("src.agents.base_agent.requests.post")
def test_planner_returns_string(mock_post):
    mock_post.return_value = make_mock("1. Search trends\n2. Analyse data")
    result = PlannerAgent().plan("What are AI trends?")
    assert isinstance(result, str) and len(result) > 0


# Critic 

@patch("src.agents.base_agent.requests.post")
def test_critic_returns_string(mock_post):
    mock_post.return_value = make_mock("Looks good.")
    result = CriticAgent().review("Sample report")
    assert isinstance(result, str) and len(result) > 0


# Analyst 

@patch("src.agents.base_agent.requests.post")
def test_analyst_returns_string(mock_post):
    mock_post.return_value = make_mock("Solar growing 20% YoY")
    docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
    result = AnalystAgent().analyze(docs, "renewable energy")
    assert isinstance(result, str) and len(result) > 0


# Researcher 

@patch("src.agents.base_agent.requests.post")
def test_researcher_extract_query(mock_post):
    mock_post.return_value = make_mock("latest renewable energy trends 2025")
    result = ResearchAgent().extract_query("1. Find trends", "energy trends?")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_researcher_search_structure(mock_post):
    mock_post.return_value = MagicMock(json=lambda: {"result": [
        {"title": "Solar Boom", "content": "Solar is growing fast.", "source": "http://example.com"}
    ]})
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[
        {"title": "Solar Boom", "content": "Solar is growing fast.", "source": "http://example.com"}
    ]):
        docs = ResearchAgent().search("solar energy trends")
    assert isinstance(docs, list)
    assert all(k in docs[0] for k in ["title", "content", "source"])

@patch("src.agents.base_agent.requests.post")
def test_mocked_base_agent(mock_post):
    mock_post.return_value = make_mock("Mocked response")
    result = PlannerAgent().plan("Test")
    assert result == "Mocked response"


# Retry logic 

def test_search_retries_on_failure():
    fake_docs = [
        {"title": "Solar", "content": "Solar is growing.", "source": "http://example.com"}
    ]
    call_count = {"n": 0}

    def flaky_call_tool(server, tool, arguments):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("rate limited")
        return fake_docs

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", side_effect=flaky_call_tool):
        docs = ResearchAgent().search("solar energy", retries=3, delay=0)
    assert len(docs) == 1


def test_search_returns_empty_after_all_retries_fail():
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", side_effect=Exception("rate limited")):
        try:
            docs = ResearchAgent().search("solar energy", retries=3, delay=0)
        except Exception:
            docs = []
    assert docs == []

# Env variable: OLLAMA_HOST 

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

# MCP client routing 

def test_mcp_client_routes_vector_store_to_8001():
    from src.mcp.client.mcp_client import MCPClient, SERVER_PORTS
    assert "8001" in SERVER_PORTS["vector_store"]

def test_mcp_client_routes_web_search_to_8002():
    from src.mcp.client.mcp_client import MCPClient, SERVER_PORTS
    assert "8002" in SERVER_PORTS["web_search"]

def test_mcp_client_call_tool_uses_correct_port():
    with patch("src.mcp.client.mcp_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"result": []})
        from src.mcp.client.mcp_client import MCPClient
        client = MCPClient()
        client.call_tool("web_search", "search", {"query": "test"})
        url_called = mock_post.call_args[0][0]
        assert "8002" in url_called

def test_mcp_client_vector_store_uses_correct_port():
    with patch("src.mcp.client.mcp_client.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"result": []})
        from src.mcp.client.mcp_client import MCPClient
        client = MCPClient()
        client.call_tool("vector_store", "search", {"query": "test"})
        url_called = mock_post.call_args[0][0]
        assert "8001" in url_called

def test_analyst_handles_mcp_unavailable():
    with patch("src.mcp.client.mcp_client.httpx.post", side_effect=Exception("connection refused")):
        with patch("src.agents.base_agent.requests.post") as mock_post:
            mock_post.return_value = make_mock("Solar is growing fast.")
            docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
            result = AnalystAgent().analyze(docs, "solar energy")
            assert isinstance(result, str) and len(result) > 0