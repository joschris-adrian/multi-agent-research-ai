import os
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gemini_response(text="gemini response"):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": text}]}}
        ]
    }
    return mock_resp


def make_ollama_response(text="ollama response"):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": text}
    return mock_resp


class TestOllamaLockedAgents:
    """
    Planner, researcher and graph builder always call Ollama
    regardless of LLM_PROVIDER.
    """

    def _make_ollama_response(self, text="ok"):
        mock = MagicMock()
        mock.json.return_value = {"response": text}
        return mock

    def test_planner_always_calls_ollama_even_when_gemini_set(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.requests.post") as mock_ollama:
                with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                    mock_ollama.return_value = self._make_ollama_response()
                    from src.agents.planner import PlannerAgent
                    with patch("src.agents.base_agent.MCPClient", MagicMock()):
                        agent = PlannerAgent()
                    agent.run("test prompt")
                    mock_ollama.assert_called_once()
                    mock_gemini.assert_not_called()

    def test_researcher_always_calls_ollama_even_when_gemini_set(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.requests.post") as mock_ollama:
                with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                    mock_ollama.return_value = self._make_ollama_response()
                    from src.agents.researcher import ResearchAgent
                    with patch("src.agents.base_agent.MCPClient", MagicMock()):
                        agent = ResearchAgent()
                    agent.run("test prompt")
                    mock_ollama.assert_called_once()
                    mock_gemini.assert_not_called()

    def test_graph_builder_always_calls_ollama_even_when_gemini_set(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.requests.post") as mock_ollama:
                with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                    mock_ollama.return_value = self._make_ollama_response()
                    from src.agents.graph_builder import GraphBuilderAgent
                    with patch("src.agents.base_agent.MCPClient", MagicMock()):
                        agent = GraphBuilderAgent()
                    agent.run("test prompt")
                    mock_ollama.assert_called_once()
                    mock_gemini.assert_not_called()

    def test_analyst_still_respects_llm_provider(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                mock_gemini.return_value = MagicMock(
                    json=lambda: {"candidates": [{"content": {"parts": [{"text": "insights"}]}}]}
                )
                mock_gemini.return_value.raise_for_status = MagicMock()
                from src.agents.analyst import AnalystAgent
                with patch("src.agents.base_agent.MCPClient", MagicMock()):
                    agent = AnalystAgent()
                agent.run("test prompt")
                mock_gemini.assert_called_once()

    def test_writer_still_respects_llm_provider(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                mock_gemini.return_value = MagicMock(
                    json=lambda: {"candidates": [{"content": {"parts": [{"text": "report"}]}}]}
                )
                mock_gemini.return_value.raise_for_status = MagicMock()
                from src.agents.writer import WriterAgent
                with patch("src.agents.base_agent.MCPClient", MagicMock()):
                    agent = WriterAgent()
                agent.run("test prompt")
                mock_gemini.assert_called_once()

    def test_critic_still_respects_llm_provider(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("src.agents.base_agent.httpx.post") as mock_gemini:
                mock_gemini.return_value = MagicMock(
                    json=lambda: {"candidates": [{"content": {"parts": [{"text": "feedback"}]}}]}
                )
                mock_gemini.return_value.raise_for_status = MagicMock()
                from src.agents.critic import CriticAgent
                with patch("src.agents.base_agent.MCPClient", MagicMock()):
                    agent = CriticAgent()
                agent.run("test prompt")
                mock_gemini.assert_called_once()

# ---------------------------------------------------------------------------
# BaseAgent routing
# ---------------------------------------------------------------------------

class TestBaseAgentRouting:

    def test_ollama_provider_calls_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("src.agents.base_agent.requests.post") as mock_post:
                mock_post.return_value = make_ollama_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test")
                result = agent.run("hello")
                mock_post.assert_called_once()
                assert result == "ollama response"

    def test_gemini_provider_calls_gemini(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                mock_post.return_value = make_gemini_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test")
                result = agent.run("hello")
                mock_post.assert_called_once()
                assert result == "gemini response"

# ---------------------------------------------------------------------------
# Gemini model assignment per agent
# ---------------------------------------------------------------------------

class TestGeminiModelAssignment:

    def _get_agent_gemini_model(self, agent_class_path, class_name):
        import importlib
        module = importlib.import_module(agent_class_path)
        cls = getattr(module, class_name)
        with patch.object(cls, "mcp", MagicMock()):
            agent = cls()
        return agent.gemini_model

    def test_analyst_uses_flash(self):
        from src.agents.analyst import AnalystAgent
        with patch("src.agents.base_agent.MCPClient", MagicMock()):
            agent = AnalystAgent()
        assert agent.gemini_model == "gemini-2.5-flash"

    def test_writer_uses_flash(self):
        from src.agents.writer import WriterAgent
        with patch("src.agents.base_agent.MCPClient", MagicMock()):
            agent = WriterAgent()
        assert agent.gemini_model == "gemini-2.5-flash"

    def test_critic_uses_flash(self):
        from src.agents.critic import CriticAgent
        with patch("src.agents.base_agent.MCPClient", MagicMock()):
            agent = CriticAgent()
        assert agent.gemini_model == "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Gemini request shape
# ---------------------------------------------------------------------------

class TestGeminiRequestShape:

    def test_system_prompt_sent_as_system_instruction(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                mock_post.return_value = make_gemini_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test goal")
                agent.run("user prompt")
                payload = mock_post.call_args.kwargs["json"]
                assert "systemInstruction" in payload
                assert "contents" in payload
                system_text = payload["systemInstruction"]["parts"][0]["text"]
                assert "Test" in system_text
                assert "Test goal" in system_text

    def test_user_prompt_in_contents(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                mock_post.return_value = make_gemini_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test")
                agent.run("specific user prompt")
                payload = mock_post.call_args.kwargs["json"]
                user_text = payload["contents"][0]["parts"][0]["text"]
                assert user_text == "specific user prompt"

    def test_temperature_passed_to_gemini(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                mock_post.return_value = make_gemini_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test", temperature=0.3)
                agent.run("prompt")
                payload = mock_post.call_args.kwargs["json"]
                assert payload["generationConfig"]["temperature"] == 0.3

    def test_correct_model_in_url(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                mock_post.return_value = make_gemini_response()
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(
                    role="Test", goal="Test",
                    gemini_model="gemini-2.0-flash"
                )
                agent.run("prompt")
                url = mock_post.call_args.args[0]
                assert "gemini-2.0-flash" in url
                assert "test-key" in url


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": ""
        }):
            from importlib import reload
            import src.agents.base_agent as module
            reload(module)
            agent = module.BaseAgent(role="Test", goal="Test")
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
                agent.run("prompt")

    def test_ollama_error_key_raises(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("src.agents.base_agent.requests.post") as mock_post:
                mock_post.return_value.json.return_value = {"error": "model not found"}
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test")
                with pytest.raises(RuntimeError, match="Ollama error"):
                    agent.run("prompt")

    def test_gemini_http_error_raises(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            with patch("src.agents.base_agent.httpx.post") as mock_post:
                import httpx
                mock_post.side_effect = httpx.HTTPStatusError(
                    "403", request=MagicMock(), response=MagicMock(status_code=403, text="forbidden")
                )
                from importlib import reload
                import src.agents.base_agent as module
                reload(module)
                agent = module.BaseAgent(role="Test", goal="Test")
                with pytest.raises(RuntimeError):
                    agent.run("prompt")


# ---------------------------------------------------------------------------
# Pipeline delay
# ---------------------------------------------------------------------------

class TestPipelineDelay:

    def _start_patches(self):
        patchers = [
            patch("src.agents.planner.PlannerAgent.plan", return_value="1. Research topic"),
            patch("src.agents.researcher.ResearchAgent.extract_query", return_value="topic"),
            patch("src.agents.researcher.ResearchAgent.search", return_value=[]),
            patch("src.agents.analyst.AnalystAgent.analyze", return_value="insights"),
            patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value={
                "companies": [], "trends": [], "technologies": [], "relationships": []
            }),
            patch("src.agents.writer.WriterAgent.write_report", return_value="report"),
            patch("src.agents.critic.CriticAgent.review", return_value="looks good"),
            patch("src.workflow.agent_pipeline.KnowledgeGraph"),
        ]
        for p in patchers:
            p.start()
        return patchers

    def test_gemini_delay_applied_between_agents(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}):
            with patch("src.workflow.agent_pipeline.time.sleep") as mock_sleep:
                patchers = self._start_patches()
                try:
                    from importlib import reload
                    import src.workflow.agent_pipeline as mod
                    reload(mod)
                    mod.MultiAgentResearchSystem().run("test question")
                finally:
                    for p in patchers:
                        p.stop()
                assert mock_sleep.call_count == 5
                expected_delay = int(os.environ.get("GEMINI_DELAY_SECONDS", "10"))
                for c in mock_sleep.call_args_list:
                    assert c.args[0] == expected_delay

    def test_ollama_no_delay_applied(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("src.workflow.agent_pipeline.time.sleep") as mock_sleep:
                patchers = self._start_patches()
                try:
                    from importlib import reload
                    import src.workflow.agent_pipeline as mod
                    reload(mod)
                    mod.MultiAgentResearchSystem().run("test question")
                finally:
                    for p in patchers:
                        p.stop()
                mock_sleep.assert_not_called()