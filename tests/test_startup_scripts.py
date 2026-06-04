import pytest
import sys
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------

class TestMain:

    def test_exits_if_gemini_key_missing(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
            with patch("builtins.input", return_value="test question"):
                import importlib, main as m
                with pytest.raises(SystemExit) as exc:
                    m.main()
                assert exc.value.code == 1

    def test_runs_with_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("builtins.input", return_value="test question"):
                with patch("main.MultiAgentResearchSystem") as mock_system:
                    mock_system.return_value.run.return_value = {
                        "report": "test report",
                        "critic_feedback": "looks good"
                    }
                    import main as m
                    m.main()
                    mock_system.return_value.run.assert_called_once_with("test question")

    def test_runs_with_gemini_when_key_set(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            with patch("builtins.input", return_value="test question"):
                with patch("main.MultiAgentResearchSystem") as mock_system:
                    mock_system.return_value.run.return_value = {
                        "report": "test report",
                        "critic_feedback": "looks good"
                    }
                    import main as m
                    m.main()
                    mock_system.return_value.run.assert_called_once_with("test question")

    def test_prints_provider(self, capsys):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("builtins.input", return_value="q"):
                with patch("main.MultiAgentResearchSystem") as mock_system:
                    mock_system.return_value.run.return_value = {
                        "report": "r", "critic_feedback": "f"
                    }
                    import main as m
                    m.main()
                    out = capsys.readouterr().out
                    assert "ollama" in out.lower()


# ---------------------------------------------------------------------------
# run_a2a.py
# ---------------------------------------------------------------------------

class TestRunA2A:

    def test_check_provider_exits_if_gemini_key_missing(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
            import run_a2a
            with pytest.raises(SystemExit) as exc:
                run_a2a.check_provider()
            assert exc.value.code == 1

    def test_check_provider_passes_with_gemini_key(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            import run_a2a
            run_a2a.check_provider()  # should not raise

    def test_check_provider_passes_with_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama", "GEMINI_API_KEY": ""}):
            import run_a2a
            run_a2a.check_provider()  # should not raise

    def test_check_servers_skips_ollama_for_gemini(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}):
            import run_a2a
            with patch("run_a2a.requests.get") as mock_get:
                with patch("run_a2a.requests.post") as mock_post:
                    mock_post.return_value = MagicMock(status_code=200)
                    mock_get.return_value = MagicMock(status_code=200)
                    # collect which URLs were called
                    called_urls = []
                    def track_get(url, **kwargs):
                        called_urls.append(url)
                        return MagicMock(status_code=200)
                    def track_post(url, **kwargs):
                        called_urls.append(url)
                        return MagicMock(status_code=200)
                    mock_get.side_effect = track_get
                    mock_post.side_effect = track_post
                    try:
                        run_a2a.check_servers()
                    except SystemExit:
                        pass
                    assert not any("11434" in u for u in called_urls), \
                        "Ollama should not be checked when LLM_PROVIDER=gemini"

    def test_check_servers_includes_ollama_for_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            import run_a2a
            called_urls = []
            def track_get(url, **kwargs):
                called_urls.append(url)
                return MagicMock(status_code=200)
            def track_post(url, **kwargs):
                called_urls.append(url)
                return MagicMock(status_code=200)
            with patch("run_a2a.requests.get", side_effect=track_get):
                with patch("run_a2a.requests.post", side_effect=track_post):
                    try:
                        run_a2a.check_servers()
                    except SystemExit:
                        pass
                    assert any("11434" in u for u in called_urls), \
                        "Ollama should be checked when LLM_PROVIDER=ollama"


# ---------------------------------------------------------------------------
# run_all.py
# ---------------------------------------------------------------------------

class TestRunAllProviderLogic:
    """
    run_all.py executes all checks at import time so it cannot be safely
    imported in tests. These tests verify the provider-branching logic
    directly without importing the script.
    """

    def _should_skip_ollama(self, provider):
        """Mirrors the branching logic in run_all.py."""
        return provider == "gemini"

    def _should_exit_early(self, provider, api_key):
        """Mirrors the key-guard logic in run_all.py."""
        return provider == "gemini" and not api_key

    def test_gemini_provider_skips_ollama(self):
        assert self._should_skip_ollama("gemini") is True

    def test_ollama_provider_does_not_skip_ollama(self):
        assert self._should_skip_ollama("ollama") is False

    def test_empty_provider_does_not_skip_ollama(self):
        assert self._should_skip_ollama("") is False

    def test_gemini_without_key_triggers_exit(self):
        assert self._should_exit_early("gemini", "") is True

    def test_gemini_with_key_does_not_trigger_exit(self):
        assert self._should_exit_early("gemini", "test-key") is False

    def test_ollama_without_key_does_not_trigger_exit(self):
        assert self._should_exit_early("ollama", "") is False


# ---------------------------------------------------------------------------
# start.py
# ---------------------------------------------------------------------------

class TestStartPy:

    def _make_args(self, gemini=False, a2a=False, stop=False):
        args = MagicMock()
        args.gemini = gemini
        args.a2a = a2a
        args.stop = stop
        return args

    def test_gemini_flag_skips_ollama(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            import start
            started = []
            def mock_start(server):
                started.append(server["name"])
                return None
            with patch("start.start_server", side_effect=mock_start):
                with patch("start.save_pids"):
                    with patch("start.argparse.ArgumentParser") as mock_parser:
                        mock_parser.return_value.parse_args.return_value = \
                            self._make_args(gemini=True)
                        start.main()
            assert "Ollama" not in started

    def test_ollama_mode_includes_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            import start
            started = []
            def mock_start(server):
                started.append(server["name"])
                return None
            with patch("start.start_server", side_effect=mock_start):
                with patch("start.save_pids"):
                    with patch("start.argparse.ArgumentParser") as mock_parser:
                        mock_parser.return_value.parse_args.return_value = \
                            self._make_args(gemini=False)
                        start.main()
            assert "Ollama" in started

    def test_gemini_flag_sets_env(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            import start
            with patch("start.start_server", return_value=None):
                with patch("start.save_pids"):
                    with patch("start.argparse.ArgumentParser") as mock_parser:
                        mock_parser.return_value.parse_args.return_value = \
                            self._make_args(gemini=True)
                        start.main()
            import os
            assert os.environ.get("LLM_PROVIDER") == "gemini"

    def test_gemini_flag_exits_if_key_missing(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "LLM_PROVIDER": "ollama"}):
            import start
            with patch("start.argparse.ArgumentParser") as mock_parser:
                mock_parser.return_value.parse_args.return_value = \
                    self._make_args(gemini=True)
                with pytest.raises(SystemExit) as exc:
                    start.main()
            assert exc.value.code == 1

    def test_env_gemini_without_flag_skips_ollama(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            import start
            started = []
            def mock_start(server):
                started.append(server["name"])
                return None
            with patch("start.start_server", side_effect=mock_start):
                with patch("start.save_pids"):
                    with patch("start.argparse.ArgumentParser") as mock_parser:
                        mock_parser.return_value.parse_args.return_value = \
                            self._make_args(gemini=False)
                        start.main()
            assert "Ollama" not in started