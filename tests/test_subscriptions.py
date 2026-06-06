import pytest
import json
import os
import time
import httpx
import importlib
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import api.main as api_module
from fastapi.testclient import TestClient
from src.scheduler.subscription_store import add_subscription, list_subscriptions, get_subscription, pause_subscription
from src.scheduler.subscription_store import resume_subscription, delete_subscription, update_last_run
from src.scheduler.scheduler import _is_due, _submitted_after, run_due_subscriptions, _submitted_after
from src.scheduler.delivery import deliver                                                                                            
from src.mcp.servers.arxiv_server import _fetch_arxiv
                        
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sub(
    topic="solar energy",
    frequency="weekly",
    delivery_method="log",
    delivery_target="",
    last_run=None,
    paused=False,
    sub_id="solar_energy_123"
):
    return {
        "id": sub_id,
        "topic": topic,
        "frequency": frequency,
        "delivery_method": delivery_method,
        "delivery_target": delivery_target,
        "last_run": last_run,
        "paused": paused,
        "created_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# subscription_store
# ---------------------------------------------------------------------------

class TestSubscriptionStore:

    def test_add_subscription_returns_dict(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            assert sub["topic"] == "solar energy"
            assert sub["frequency"] == "weekly"
            assert sub["paused"] is False
            assert sub["last_run"] is None
            assert "id" in sub

    def test_add_subscription_persists_to_file(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            add_subscription("solar energy", "weekly", "log", "")
            with open(store_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["topic"] == "solar energy"

    def test_list_subscriptions_returns_all(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            add_subscription("solar energy", "weekly", "log", "")
            add_subscription("AI agents", "daily", "log", "")
            subs = list_subscriptions()
            assert len(subs) == 2

    def test_list_subscriptions_empty_when_no_file(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            assert list_subscriptions() == []

    def test_get_subscription_returns_correct_one(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            result = get_subscription(sub["id"])
            assert result["topic"] == "solar energy"

    def test_get_subscription_returns_none_for_unknown_id(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            assert get_subscription("nonexistent") is None

    def test_pause_subscription(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            ok = pause_subscription(sub["id"])
            assert ok is True
            assert get_subscription(sub["id"])["paused"] is True

    def test_pause_unknown_subscription_returns_false(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            assert pause_subscription("nonexistent") is False

    def test_resume_subscription(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            pause_subscription(sub["id"])
            ok = resume_subscription(sub["id"])
            assert ok is True
            assert get_subscription(sub["id"])["paused"] is False

    def test_delete_subscription(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            ok = delete_subscription(sub["id"])
            assert ok is True
            assert list_subscriptions() == []

    def test_delete_unknown_subscription_returns_false(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            assert delete_subscription("nonexistent") is False

    def test_update_last_run(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            assert sub["last_run"] is None
            update_last_run(sub["id"])
            updated = get_subscription(sub["id"])
            assert updated["last_run"] is not None

    def test_subscription_id_contains_topic_slug(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub = add_subscription("solar energy", "weekly", "log", "")
            assert "solar_energy" in sub["id"]

    def test_multiple_subscriptions_have_unique_ids(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            sub1 = add_subscription("solar energy", "weekly", "log", "")
            time.sleep(0.01)
            sub2 = add_subscription("solar energy", "weekly", "log", "")
            assert sub1["id"] != sub2["id"]


# ---------------------------------------------------------------------------
# scheduler — due logic
# ---------------------------------------------------------------------------

class TestSchedulerDueLogic:

    def test_no_last_run_is_due(self):
        sub = make_sub(last_run=None, paused=False)
        assert _is_due(sub) is True

    def test_paused_is_never_due(self):
        sub = make_sub(last_run=None, paused=True)
        assert _is_due(sub) is False

    def test_weekly_due_after_7_days(self):
        last = (datetime.utcnow() - timedelta(days=8)).isoformat()
        sub = make_sub(last_run=last, frequency="weekly")
        assert _is_due(sub) is True

    def test_weekly_not_due_before_7_days(self):
        last = (datetime.utcnow() - timedelta(days=3)).isoformat()
        sub = make_sub(last_run=last, frequency="weekly")
        assert _is_due(sub) is False

    def test_daily_due_after_1_day(self):
        last = (datetime.utcnow() - timedelta(days=2)).isoformat()
        sub = make_sub(last_run=last, frequency="daily")
        assert _is_due(sub) is True

    def test_daily_not_due_before_1_day(self):
        last = (datetime.utcnow() - timedelta(hours=12)).isoformat()
        sub = make_sub(last_run=last, frequency="daily")
        assert _is_due(sub) is False

    def test_monthly_due_after_30_days(self):
        last = (datetime.utcnow() - timedelta(days=31)).isoformat()
        sub = make_sub(last_run=last, frequency="monthly")
        assert _is_due(sub) is True

    def test_monthly_not_due_before_30_days(self):
        last = (datetime.utcnow() - timedelta(days=15)).isoformat()
        sub = make_sub(last_run=last, frequency="monthly")
        assert _is_due(sub) is False

    def test_unknown_frequency_defaults_to_weekly(self):
        last = (datetime.utcnow() - timedelta(days=8)).isoformat()
        sub = make_sub(last_run=last, frequency="fortnightly")
        assert _is_due(sub) is True

    def test_submitted_after_returns_empty_when_no_last_run(self):
        sub = make_sub(last_run=None)
        assert _submitted_after(sub) == ""

    def test_submitted_after_returns_date_portion(self):
        sub = make_sub(last_run="2024-06-01T12:00:00")
        assert _submitted_after(sub) == "2024-06-01"


# ---------------------------------------------------------------------------
# scheduler — run_due_subscriptions
# ---------------------------------------------------------------------------

class TestRunDueSubscriptions:

    def test_runs_pipeline_for_due_subscription(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        sub = make_sub(last_run=None)
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path):
            with patch("src.scheduler.scheduler.list_subscriptions", return_value=[sub]):
                with patch("src.scheduler.scheduler.update_last_run") as mock_update:
                    with patch("src.scheduler.scheduler.deliver") as mock_deliver:
                        with patch("src.scheduler.scheduler.MultiAgentResearchSystem") as mock_system:
                            mock_system.return_value.run.return_value = {
                                "report": "test report",
                                "tasks": "", "documents": [],
                                "insights": "", "entities": {}, "critic_feedback": ""
                            }
                            run_due_subscriptions()
                            mock_system.return_value.run.assert_called_once_with(sub["topic"])
                            mock_update.assert_called_once_with(sub["id"])
                            mock_deliver.assert_called_once()

    def test_skips_paused_subscription(self):
        sub = make_sub(paused=True)
        with patch("src.scheduler.scheduler.list_subscriptions", return_value=[sub]):
            with patch("src.scheduler.scheduler.MultiAgentResearchSystem") as mock_system:
                run_due_subscriptions()
                mock_system.return_value.run.assert_not_called()

    def test_skips_subscription_not_due(self):
        last = (datetime.utcnow() - timedelta(days=1)).isoformat()
        sub = make_sub(last_run=last, frequency="weekly")
        with patch("src.scheduler.scheduler.list_subscriptions", return_value=[sub]):
            with patch("src.scheduler.scheduler.MultiAgentResearchSystem") as mock_system:
                run_due_subscriptions()
                mock_system.return_value.run.assert_not_called()

    def test_handles_pipeline_failure_gracefully(self):
        sub = make_sub(last_run=None)
        with patch("src.scheduler.scheduler.list_subscriptions", return_value=[sub]):
            with patch("src.scheduler.scheduler.MultiAgentResearchSystem") as mock_system:
                mock_system.return_value.run.side_effect = Exception("pipeline error")
                with patch("src.scheduler.scheduler.update_last_run") as mock_update:
                    run_due_subscriptions()
                    mock_update.assert_not_called()

    def test_runs_multiple_due_subscriptions(self):
        sub1 = make_sub(topic="solar energy", sub_id="sub1")
        sub2 = make_sub(topic="AI agents", sub_id="sub2")
        with patch("src.scheduler.scheduler.list_subscriptions", return_value=[sub1, sub2]):
            with patch("src.scheduler.scheduler.update_last_run"):
                with patch("src.scheduler.scheduler.deliver"):
                    with patch("src.scheduler.scheduler.MultiAgentResearchSystem") as mock_system:
                        mock_system.return_value.run.return_value = {
                            "report": "r", "tasks": "", "documents": [],
                            "insights": "", "entities": {}, "critic_feedback": ""
                        }
                        run_due_subscriptions()
                        assert mock_system.return_value.run.call_count == 2

    def test_no_subscriptions_does_not_crash(self):
        with patch("src.scheduler.scheduler.list_subscriptions", return_value=[]):
            run_due_subscriptions()


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------

class TestDelivery:

    def test_log_delivery_does_not_raise(self, capsys):
        deliver("test report", "solar energy", "log", "")
        out = capsys.readouterr().out
        assert "solar energy" in out

    def test_slack_webhook_called(self):
        with patch("src.scheduler.delivery.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            deliver("test report", "solar energy", "slack", "https://hooks.slack.com/test")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "solar energy" in payload["text"]

    def test_discord_webhook_called(self):
        with patch("src.scheduler.delivery.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            deliver("test report", "solar energy", "discord", "https://discord.com/api/webhooks/test")
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "solar energy" in payload["content"]

    def test_slack_report_truncated_to_2000_chars(self):
        with patch("src.scheduler.delivery.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(raise_for_status=MagicMock())
            long_report = "x" * 5000
            deliver(long_report, "topic", "slack", "https://hooks.slack.com/test")
            payload = mock_post.call_args.kwargs["json"]
            assert len(payload["text"]) <= 2100

    def test_email_skips_if_smtp_not_configured(self, capsys):
        with patch.dict("os.environ", {"SMTP_HOST": "", "SMTP_USER": "", "SMTP_PASSWORD": ""}):
            deliver("test report", "solar energy", "email", "test@example.com")
            out = capsys.readouterr().out
            assert "SMTP not configured" in out

    def test_email_sends_when_smtp_configured(self):
        with patch.dict("os.environ", {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_USER": "user@example.com",
            "SMTP_PASSWORD": "password"
        }):
            with patch("src.scheduler.delivery.smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
                mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
                deliver("test report", "solar energy", "email", "recipient@example.com")
                mock_server.sendmail.assert_called_once()

    def test_unknown_delivery_method_does_not_raise(self, capsys):
        deliver("test report", "solar energy", "carrier_pigeon", "")
        out = capsys.readouterr().out
        assert "unknown method" in out

    def test_webhook_failure_does_not_raise(self):
        with patch("src.scheduler.delivery.httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("timeout")
            deliver("test report", "solar energy", "slack", "https://hooks.slack.com/test")


# ---------------------------------------------------------------------------
# arxiv_server — submitted_after parameter
# ---------------------------------------------------------------------------

class TestArxivSubmittedAfter:

    def test_submitted_after_added_to_query(self):
        with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                text="""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
            )
            _fetch_arxiv("solar energy", 5, submitted_after="2024-01-01")
            url = mock_get.call_args.args[0]
            assert "submittedDate" in url
            assert "20240101" in url

    def test_no_submitted_after_omits_date_filter(self):
        with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                text="""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
            )
            _fetch_arxiv("solar energy", 5, submitted_after="")
            url = mock_get.call_args.args[0]
            # sortBy=submittedDate is always present as a sort param
            # the date range filter only appears in search_query when submitted_after is set
            from urllib.parse import unquote
            decoded = unquote(url)
            assert "submittedDate:[" not in decoded

    def test_submitted_after_strips_hyphens(self):
        with patch("src.mcp.servers.arxiv_server.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                text="""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
            )
            _fetch_arxiv("solar energy", 5, submitted_after="2024-06-15")
            url = mock_get.call_args.args[0]
            assert "20240615" in url


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class TestSubscriptionAPI:

    @pytest.fixture
    def client(self, tmp_path):
        store_path = str(tmp_path / "subscriptions.json")
        # patch the constant in every function that uses it directly
        with patch("src.scheduler.subscription_store.STORE_PATH", store_path), \
             patch("src.scheduler.scheduler.list_subscriptions",
                   wraps=lambda: _patched_list(store_path)), \
             patch("src.scheduler.scheduler.update_last_run",
                   wraps=lambda sid: None):
            from api.main import app
            yield TestClient(app)

    def test_create_subscription(self, client):
        response = client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        assert response.status_code == 200
        data = response.json()
        assert data["topic"] == "solar energy"
        assert "id" in data

    def test_list_subscriptions_empty(self, client):
        response = client.get("/subscriptions")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_subscriptions_after_create(self, client):
        client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        response = client.get("/subscriptions")
        assert len(response.json()) == 1

    def test_get_subscription_by_id(self, client):
        create = client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        sub_id = create.json()["id"]
        response = client.get(f"/subscriptions/{sub_id}")
        assert response.status_code == 200
        assert response.json()["topic"] == "solar energy"

    def test_get_unknown_subscription_returns_404(self, client):
        response = client.get("/subscriptions/nonexistent")
        assert response.status_code == 404

    def test_pause_subscription(self, client):
        create = client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        sub_id = create.json()["id"]
        response = client.post(f"/subscriptions/{sub_id}/pause")
        assert response.status_code == 200
        assert response.json()["status"] == "paused"

    def test_resume_subscription(self, client):
        create = client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        sub_id = create.json()["id"]
        client.post(f"/subscriptions/{sub_id}/pause")
        response = client.post(f"/subscriptions/{sub_id}/resume")
        assert response.status_code == 200
        assert response.json()["status"] == "resumed"

    def test_delete_subscription(self, client):
        create = client.post("/subscriptions", json={
            "topic": "solar energy",
            "frequency": "weekly",
            "delivery_method": "log",
            "delivery_target": ""
        })
        sub_id = create.json()["id"]
        response = client.delete(f"/subscriptions/{sub_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_unknown_returns_404(self, client):
        response = client.delete("/subscriptions/nonexistent")
        assert response.status_code == 404

    def test_run_subscriptions_endpoint(self, client):
        with patch("api.main.run_due_subscriptions"):
            response = client.post("/subscriptions/run")
            assert response.status_code == 200
            assert response.json()["status"] == "scheduler triggered"