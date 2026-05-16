import sys
import pytest
from unittest.mock import patch, MagicMock


# check_servers 

def test_check_servers_passes_when_both_reachable():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"result": []})
        from scripts.supercharge import check_servers
        try:
            check_servers()
        except SystemExit:
            assert False, "check_servers should not exit when servers are reachable"


def test_check_servers_exits_when_vector_store_unreachable():
    with patch("scripts.supercharge.httpx.post", side_effect=Exception("connection refused")):
        from scripts.supercharge import check_servers
        with pytest.raises(SystemExit):
            check_servers()


# fetch_documents 

def test_fetch_documents_returns_list():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": [
                {"title": "Solar", "content": "Solar is growing.", "source": "http://example.com"}
            ]}
        )
        from scripts.supercharge import fetch_documents
        docs = fetch_documents("solar energy", max_results=2)
        assert isinstance(docs, list)
        assert len(docs) > 0


def test_fetch_documents_deduplicates_by_source():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": [
                {"title": "Solar", "content": "Solar is growing.", "source": "http://example.com"}
            ]}
        )
        from scripts.supercharge import fetch_documents
        docs = fetch_documents("solar energy", max_results=5)
        sources = [d["source"] for d in docs]
        assert len(sources) == len(set(sources))


def test_fetch_documents_uses_multiple_queries():
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        return MagicMock(status_code=200, json=lambda: {"result": []})

    with patch("scripts.supercharge.httpx.post", side_effect=fake_post):
        from scripts.supercharge import fetch_documents
        fetch_documents("solar energy", max_results=2)
        assert call_count["n"] > 1


def test_fetch_documents_handles_failed_query_gracefully():
    with patch("scripts.supercharge.httpx.post", side_effect=Exception("rate limited")):
        from scripts.supercharge import fetch_documents
        docs = fetch_documents("solar energy", max_results=2)
        assert docs == []


def test_fetch_documents_all_have_required_keys():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": [
                {"title": "Solar", "content": "Growing fast.", "source": "http://example.com"}
            ]}
        )
        from scripts.supercharge import fetch_documents
        docs = fetch_documents("solar energy", max_results=2)
        assert all(k in doc for doc in docs for k in ["title", "content", "source"])


# store_documents 

def test_store_documents_posts_to_vector_store():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        from scripts.supercharge import store_documents
        docs = [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        store_documents(docs)
        url_called = mock_post.call_args[0][0]
        assert "8001" in url_called
        assert "vector_store/add" in url_called


def test_store_documents_exits_on_failure():
    with patch("scripts.supercharge.httpx.post", side_effect=Exception("connection refused")):
        from scripts.supercharge import store_documents
        with pytest.raises(SystemExit):
            store_documents([{"title": "Solar", "content": "Growing.", "source": "http://example.com"}])


def test_store_documents_sends_correct_payload():
    with patch("scripts.supercharge.httpx.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        from scripts.supercharge import store_documents
        docs = [{"title": "Solar", "content": "Growing.", "source": "http://example.com"}]
        store_documents(docs)
        payload = mock_post.call_args[1]["json"]
        assert "documents" in payload
        assert payload["documents"] == docs