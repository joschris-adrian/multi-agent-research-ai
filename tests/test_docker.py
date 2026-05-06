"""
Docker integration tests.
These tests verify Docker configuration is correct without needing
Docker to actually be running. Run live tests separately with:
    docker-compose up --build
    pytest tests/test_docker.py --docker
"""
import os
import yaml
import pytest


COMPOSE_FILE = "docker-compose.yml"


# docker-compose.yml structure 

def test_compose_file_exists():
    assert os.path.exists(COMPOSE_FILE), "docker-compose.yml not found in project root"


def test_compose_has_required_services():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    services = compose.get("services", {})
    assert "ollama" in services
    assert "api" in services
    assert "ui" in services


def test_compose_api_depends_on_ollama():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    api = compose["services"]["api"]
    assert "ollama" in api.get("depends_on", [])


def test_compose_ui_depends_on_api():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    ui = compose["services"]["ui"]
    assert "api" in ui.get("depends_on", [])


def test_compose_correct_ports():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    services = compose["services"]
    assert any("11434" in str(p) for p in services["ollama"].get("ports", []))
    assert any("8000" in str(p) for p in services["api"].get("ports", []))
    assert any("8501" in str(p) for p in services["ui"].get("ports", []))


def test_compose_env_variables_set():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    api_env = compose["services"]["api"].get("environment", [])
    ui_env = compose["services"]["ui"].get("environment", [])
    assert any("OLLAMA_HOST" in str(e) for e in api_env)
    assert any("API_URL" in str(e) for e in ui_env)


def test_compose_ollama_volume_persists():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    ollama = compose["services"]["ollama"]
    assert "volumes" in ollama
    assert "ollama_data" in compose.get("volumes", {})


# Dockerfile existence 

def test_dockerfile_api_exists():
    assert os.path.exists("Dockerfile.api"), "Dockerfile.api not found"


def test_dockerfile_ui_exists():
    assert os.path.exists("Dockerfile.ui"), "Dockerfile.ui not found"


# Env variable defaults 

def test_ollama_host_default():
    os.environ.pop("OLLAMA_HOST", None)
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    assert host == "http://localhost:11434"


def test_api_url_default():
    os.environ.pop("API_URL", None)
    url = os.getenv("API_URL", "http://localhost:8000")
    assert url == "http://localhost:8000"


def test_ollama_host_docker_override():
    with patch_env("OLLAMA_HOST", "http://ollama:11434"):
        assert os.getenv("OLLAMA_HOST") == "http://ollama:11434"


def test_api_url_docker_override():
    with patch_env("API_URL", "http://api:8000"):
        assert os.getenv("API_URL") == "http://api:8000"

def test_compose_has_mcp_server_services():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    services = compose.get("services", {})
    assert "mcp_vector_store" in services, "mcp_vector_store service not found in docker-compose.yml"
    assert "mcp_web_search" in services, "mcp_web_search service not found in docker-compose.yml"


def test_compose_mcp_correct_ports():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    vs = compose["services"].get("mcp_vector_store", {})
    ws = compose["services"].get("mcp_web_search", {})
    assert any("8001" in str(p) for p in vs.get("ports", []))
    assert any("8002" in str(p) for p in ws.get("ports", []))


def test_mcp_host_default():
    os.environ.pop("MCP_HOST", None)
    host = os.getenv("MCP_HOST", "http://localhost:8001")
    assert host == "http://localhost:8001"


def test_compose_mcp_depends_on_ollama():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    vs = compose["services"].get("mcp_vector_store", {})
    ws = compose["services"].get("mcp_web_search", {})
    assert "ollama" in vs.get("depends_on", [])
    assert "ollama" in ws.get("depends_on", [])


def test_compose_mcp_env_has_mcp_host():
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)
    vs_env = compose["services"].get("mcp_vector_store", {}).get("environment", [])
    ws_env = compose["services"].get("mcp_web_search", {}).get("environment", [])
    assert any("MCP_HOST" in str(e) for e in vs_env)
    assert any("MCP_HOST" in str(e) for e in ws_env)
# Helper 

from contextlib import contextmanager

@contextmanager
def patch_env(key, value):
    os.environ[key] = value
    try:
        yield
    finally:
        os.environ.pop(key, None)
