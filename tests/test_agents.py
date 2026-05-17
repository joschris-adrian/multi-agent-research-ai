import os
import time
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
def test_mocked_base_agent(mock_post):
    mock_post.return_value = make_mock("Mocked response")
    result = PlannerAgent().plan("Test")
    assert result == "Mocked response"


# Retry logic 

def test_search_retries_on_failure():
    fake_docs = [
        {"title": "Solar", "content": "Solar is growing.", "source": "http://example.com"}
    ]
    web_call_count = {"n": 0}

    def flaky_call_tool(server, tool, arguments):
        if server == "web_search":
            web_call_count["n"] += 1
            if web_call_count["n"] < 3:
                raise Exception("rate limited")
            return fake_docs
        if server == "arxiv":
            return []
        return []

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

# Chunking 

def test_researcher_chunks_long_content():
    long_body = "Solar energy word. " * 60
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Solar", "body": long_body, "href": "http://example.com"}
        ]
        from src.mcp.servers.web_search_server import search, SearchRequest
        req = SearchRequest(query="solar", max_results=1, retries=1, delay=0)
        result = search(req)
        docs = result["result"]
        assert len(docs) > 1


def test_researcher_chunks_short_content_stays_single():
    short_body = "Solar is growing fast."
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Solar", "body": short_body, "href": "http://example.com"}
        ]
        from src.mcp.servers.web_search_server import search, SearchRequest
        req = SearchRequest(query="solar", max_results=1, retries=1, delay=0)
        result = search(req)
        docs = result["result"]
        assert len(docs) == 1


def test_researcher_empty_chunks_are_excluded():
    body = "Solar. " + " " * 300 + "Wind."
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Energy", "body": body, "href": "http://example.com"}
        ]
        from src.mcp.servers.web_search_server import search, SearchRequest
        req = SearchRequest(query="energy", max_results=1, retries=1, delay=0)
        result = search(req)
        docs = result["result"]
        assert all(d["content"].strip() for d in docs)


def test_researcher_chunks_preserve_title_and_source():
    long_body = "Renewable energy content. " * 50
    with patch("src.mcp.servers.web_search_server.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Energy", "body": long_body, "href": "http://example.com"}
        ]
        from src.mcp.servers.web_search_server import search, SearchRequest
        req = SearchRequest(query="energy", max_results=1, retries=1, delay=0)
        result = search(req)
        docs = result["result"]
        assert all(d["title"] == "Energy" for d in docs)
        assert all(d["source"] == "http://example.com" for d in docs)


# RAG retrieval 

def test_analyst_passes_top_k_to_mcp():
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]) as mock_tool:
        with patch("src.agents.base_agent.requests.post") as mock_post:
            mock_post.return_value = make_mock("Solar insights")
            docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
            AnalystAgent().analyze(docs, "solar energy")
            search_calls = [c for c in mock_tool.call_args_list if c[0][1] == "search"]
            assert len(search_calls) > 0
            assert search_calls[0][0][2].get("top_k") == 5


def test_analyst_prompt_does_not_include_relevance_score():
    scored_docs = [{"content": "Solar grew 40%.", "score": 0.85}]
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("insights")

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=scored_docs):
        with patch("src.agents.base_agent.requests.post") as mock_post:
            mock_post.side_effect = capture
            docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
            AnalystAgent().analyze(docs, "solar energy")
            assert "0.85" not in captured["prompt"]
            assert "Solar grew 40%" in captured["prompt"]

def test_analyst_prompt_uses_rag_framing():
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("insights")

    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        with patch("src.agents.base_agent.requests.post") as mock_post:
            mock_post.side_effect = capture
            docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
            AnalystAgent().analyze(docs, "solar energy")
            assert "retrieved from memory" in captured["prompt"].lower()

# Prompt engineering - Planner 

@patch("src.agents.base_agent.requests.post")
def test_planner_prompt_includes_few_shot_example(mock_post):
    mock_post.return_value = make_mock("1. Search trends\n2. Analyse data")
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("1. Search trends")

    mock_post.side_effect = capture
    PlannerAgent().plan("What are AI trends?")
    assert "electric vehicles" in captured["prompt"].lower()
    assert "1." in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_planner_prompt_asks_for_numbered_list(mock_post):
    mock_post.return_value = make_mock("1. Search trends")
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("1. Search trends")

    mock_post.side_effect = capture
    PlannerAgent().plan("What are AI trends?")
    assert "numbered" in captured["prompt"].lower()


# Prompt engineering - Analyst 

@patch("src.agents.base_agent.requests.post")
def test_analyst_prompt_includes_chain_of_thought(mock_post):
    mock_post.return_value = make_mock("Solar growing 20% YoY")
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("Solar growing 20% YoY")

    mock_post.side_effect = capture
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
        AnalystAgent().analyze(docs, "solar energy")
    assert "step by step" in captured["prompt"].lower()
    assert "1." in captured["prompt"]
    assert "2." in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_analyst_prompt_includes_current_research(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("insights")

    mock_post.side_effect = capture
    with patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]):
        docs = [{"title": "Solar", "content": "Solar is booming.", "source": "http://example.com"}]
        AnalystAgent().analyze(docs, "solar energy")
    assert "Solar is booming" in captured["prompt"]


# Prompt engineering - Writer 

@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_includes_output_schema(mock_post):
    mock_post.return_value = make_mock("# Report")
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing.", FAKE_ENTITIES)
    assert "## Summary" in captured["prompt"]
    assert "## Key Trends" in captured["prompt"]
    assert "## Key Players" in captured["prompt"]
    assert "## Statistics" in captured["prompt"]
    assert "## Conclusion" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_references_knowledge_graph(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing.", FAKE_ENTITIES)
    assert "knowledge graph" in captured["prompt"].lower()


# Prompt engineering - Critic 

@patch("src.agents.base_agent.requests.post")
def test_critic_prompt_is_adversarial(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("Looks good.")

    mock_post.side_effect = capture
    CriticAgent().review("Sample report")
    assert "adversarial" in captured["prompt"].lower() or "weaknesses" in captured["prompt"].lower()


@patch("src.agents.base_agent.requests.post")
def test_critic_prompt_checks_for_unsupported_claims(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("Looks good.")

    mock_post.side_effect = capture
    CriticAgent().review("Sample report")
    assert "evidence" in captured["prompt"].lower()
    assert "sources" in captured["prompt"].lower()


# Prompt engineering - Graph Builder 

@patch("src.agents.base_agent.requests.post")
def test_graph_builder_prompt_includes_negative_rules(mock_post):
    fake_entities = {
        "companies": ["Tesla"],
        "trends": [],
        "technologies": [],
        "relationships": []
    }
    import json
    mock_post.return_value = make_mock(json.dumps(fake_entities))
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(json.dumps(fake_entities))

    mock_post.side_effect = capture
    GraphBuilderAgent().extract_entities("Tesla uses batteries.", "EVs")
    assert "do not invent" in captured["prompt"].lower() or "do not include" in captured["prompt"].lower()
    assert "json" in captured["prompt"].lower()


@patch("src.agents.base_agent.requests.post")
def test_graph_builder_prompt_forbids_generic_terms(mock_post):
    fake_entities = {"companies": [], "trends": [], "technologies": [], "relationships": []}
    import json
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(json.dumps(fake_entities))

    mock_post.side_effect = capture
    GraphBuilderAgent().extract_entities("Some insights.", "topic")
    assert "industry" in captured["prompt"].lower() or "generic" in captured["prompt"].lower()


# System prompt 

@patch("src.agents.base_agent.requests.post")
def test_system_prompt_includes_pipeline_context(mock_post):
    mock_post.return_value = make_mock("ok")
    agent = BaseAgent(role="Tester", goal="Test")
    agent.run("hello")
    system = mock_post.call_args[1]["json"]["system"]
    assert "pipeline" in system.lower()


@patch("src.agents.base_agent.requests.post")
def test_system_prompt_instructs_not_to_guess(mock_post):
    mock_post.return_value = make_mock("ok")
    agent = BaseAgent(role="Tester", goal="Test")
    agent.run("hello")
    system = mock_post.call_args[1]["json"]["system"]
    assert "guessing" in system.lower() or "unavailable" in system.lower()

# TTL / vector store eviction 

def test_vector_store_evicts_expired_documents():
    from src.memory.vector_store import VectorStore, TTL_SECONDS
    import time

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        old_timestamp = int(time.time()) - TTL_SECONDS - 100
        mock_collection.get.return_value = {
            "ids": ["doc_1", "doc_2"],
            "metadatas": [
                {"timestamp": old_timestamp},
                {"timestamp": int(time.time())}
            ]
        }

        store = VectorStore()
        store.evict_expired()
        mock_collection.delete.assert_called_once_with(ids=["doc_1"])


def test_vector_store_does_not_evict_fresh_documents():
    from src.memory.vector_store import VectorStore

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.get.return_value = {
            "ids": ["doc_1"],
            "metadatas": [{"timestamp": int(time.time())}]
        }

        store = VectorStore()
        store.evict_expired()
        mock_collection.delete.assert_not_called()


def test_vector_store_add_stores_timestamp():
    from src.memory.vector_store import VectorStore
    import time

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore()
        store.add_documents([{
            "title": "Solar",
            "content": "Solar is growing.",
            "source": "http://example.com"
        }])

        call_kwargs = mock_collection.add.call_args[1]
        assert "metadatas" in call_kwargs
        assert "timestamp" in call_kwargs["metadatas"][0]
        assert abs(call_kwargs["metadatas"][0]["timestamp"] - int(time.time())) < 5


def test_vector_store_eviction_handles_missing_timestamp():
    from src.memory.vector_store import VectorStore

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.get.return_value = {
            "ids": ["doc_1"],
            "metadatas": [{}]
        }

        store = VectorStore()
        store.evict_expired()
        mock_collection.delete.assert_not_called()


def test_vector_store_eviction_handles_failure_gracefully():
    from src.memory.vector_store import VectorStore

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.side_effect = Exception("db error")
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore()
        try:
            store.evict_expired()
        except Exception:
            assert False, "evict_expired should not raise"


def test_vector_store_search_triggers_eviction():
    from src.memory.vector_store import VectorStore

    with patch("src.memory.vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_collection.query.return_value = {
            "documents": [[]], "distances": [[]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore()
        store.search("solar energy")
        mock_collection.get.assert_called_once()

# Writer key players prompt constraints 

@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_instructs_no_placeholder_companies(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing rapidly.", FAKE_ENTITIES)
    assert "placeholder" in captured["prompt"].lower() or "no specific" in captured["prompt"].lower()


@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_requires_context_for_entities(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing rapidly.", FAKE_ENTITIES)
    assert "supporting context" in captured["prompt"].lower() or "specific detail" in captured["prompt"].lower()


@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_instructs_fallback_when_no_orgs(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing rapidly.", {})
    assert "no specific" in captured["prompt"].lower() or "not identified" in captured["prompt"].lower()


@patch("src.agents.base_agent.requests.post")
def test_writer_prompt_warns_against_entities_without_detail(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("# Report")

    mock_post.side_effect = capture
    WriterAgent().write_report("Solar is growing rapidly.", FAKE_ENTITIES)
    assert "knowledge graph" in captured["prompt"].lower()
    assert "supporting" in captured["prompt"].lower() or "context" in captured["prompt"].lower()