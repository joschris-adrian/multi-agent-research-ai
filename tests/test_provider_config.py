import pytest
import os
import json
from unittest.mock import patch, MagicMock


ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


# ---------------------------------------------------------------------------
# load_provider_config
# ---------------------------------------------------------------------------

class TestLoadProviderConfig:

    def test_returns_ollama_when_not_set(self):
        with patch("ui.provider_config.dotenv_values", return_value={}):
            from ui.provider_config import load_provider_config
            config = load_provider_config()
            assert config["provider"] == "ollama"
            assert config["api_key"] == ""

    def test_returns_gemini_when_set(self):
        with patch("ui.provider_config.dotenv_values", return_value={
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key"
        }):
            from ui.provider_config import load_provider_config
            config = load_provider_config()
            assert config["provider"] == "gemini"
            assert config["api_key"] == "test-key"

    def test_lowercases_provider(self):
        with patch("ui.provider_config.dotenv_values", return_value={
            "LLM_PROVIDER": "GEMINI"
        }):
            from ui.provider_config import load_provider_config
            config = load_provider_config()
            assert config["provider"] == "gemini"

    def test_returns_empty_key_when_not_set(self):
        with patch("ui.provider_config.dotenv_values", return_value={
            "LLM_PROVIDER": "gemini"
        }):
            from ui.provider_config import load_provider_config
            config = load_provider_config()
            assert config["api_key"] == ""


# ---------------------------------------------------------------------------
# save_provider_config
# ---------------------------------------------------------------------------

class TestSaveProviderConfig:

    def test_saves_ollama_provider(self):
        with patch("ui.provider_config.set_key") as mock_set:
            from ui.provider_config import save_provider_config
            save_provider_config("ollama")
            assert mock_set.call_count == 1
            call_args = mock_set.call_args.args
            assert call_args[1] == "LLM_PROVIDER"
            assert call_args[2] == "ollama"
            assert call_args[0].endswith(".env")

    def test_saves_gemini_provider_and_key(self):
        with patch("ui.provider_config.set_key") as mock_set:
            from ui.provider_config import save_provider_config
            save_provider_config("gemini", "my-api-key")
            calls = [str(c) for c in mock_set.call_args_list]
            assert any("LLM_PROVIDER" in c for c in calls)
            assert any("GEMINI_API_KEY" in c for c in calls)
            assert any("my-api-key" in c for c in calls)

    def test_does_not_save_empty_key_for_gemini(self):
        with patch("ui.provider_config.set_key") as mock_set:
            from ui.provider_config import save_provider_config
            save_provider_config("gemini", "")
            calls = [str(c) for c in mock_set.call_args_list]
            assert not any("GEMINI_API_KEY" in c for c in calls)

    def test_does_not_save_key_for_ollama(self):
        with patch("ui.provider_config.set_key") as mock_set:
            from ui.provider_config import save_provider_config
            save_provider_config("ollama", "some-key")
            calls = [str(c) for c in mock_set.call_args_list]
            assert not any("GEMINI_API_KEY" in c for c in calls)


# ---------------------------------------------------------------------------
# validate_gemini_key
# ---------------------------------------------------------------------------

class TestValidateGeminiKey:

    def _make_success_response(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}]
        }
        mock.raise_for_status = MagicMock()
        return mock

    def _make_error_response(self, status_code, text):
        import httpx
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text
        return httpx.HTTPStatusError(
            str(status_code),
            request=mock_request,
            response=mock_response
        )

    def test_valid_key_returns_true(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.return_value = self._make_success_response()
            from ui.provider_config import validate_gemini_key
            valid, error = validate_gemini_key("test-key")
            assert valid is True
            assert error == ""

    def test_invalid_key_returns_false_with_message(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.side_effect = self._make_error_response(403, "forbidden")
            from ui.provider_config import validate_gemini_key
            valid, error = validate_gemini_key("bad-key")
            assert valid is False
            assert "403" in error

    def test_rate_limited_returns_false_with_message(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.side_effect = self._make_error_response(429, "quota exceeded")
            from ui.provider_config import validate_gemini_key
            valid, error = validate_gemini_key("test-key")
            assert valid is False
            assert "429" in error

    def test_network_error_returns_false(self):
        import httpx
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("timeout")
            from ui.provider_config import validate_gemini_key
            valid, error = validate_gemini_key("test-key")
            assert valid is False
            assert "Network error" in error

    def test_correct_model_used_in_url(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.return_value = self._make_success_response()
            from ui.provider_config import validate_gemini_key
            validate_gemini_key("test-key", model="gemini-2.5-flash")
            url = mock_post.call_args.args[0]
            assert "gemini-2.5-flash" in url

    def test_api_key_in_url(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.return_value = self._make_success_response()
            from ui.provider_config import validate_gemini_key
            validate_gemini_key("my-secret-key")
            url = mock_post.call_args.args[0]
            assert "my-secret-key" in url

    def test_empty_key_still_calls_api(self):
        with patch("ui.provider_config.httpx.post") as mock_post:
            mock_post.side_effect = self._make_error_response(400, "bad request")
            from ui.provider_config import validate_gemini_key
            valid, error = validate_gemini_key("")
            assert valid is False