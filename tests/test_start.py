import json
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from start import is_port_in_use, check_health, save_pids, load_pids, start_server, stop_servers, SERVERS, main

# is_port_in_use 

def test_is_port_in_use_returns_true_when_port_open():
    with patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0
        assert is_port_in_use(8001) is True


def test_is_port_in_use_returns_false_when_port_closed():
    with patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 1
        assert is_port_in_use(8001) is False


# check_health 

def test_check_health_returns_true_on_200_get():
    with patch("start.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        server = {
            "health_url": "http://localhost:11434",
            "health_method": "GET",
            "health_payload": None
        }
        assert check_health(server) is True


def test_check_health_returns_true_on_200_post():
    with patch("start.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        server = {
            "health_url": "http://localhost:8001/vector_store/search",
            "health_method": "POST",
            "health_payload": {"query": "health"}
        }
        assert check_health(server) is True


def test_check_health_returns_false_on_connection_error():
    with patch("start.requests.get", side_effect=Exception("connection refused")):
        server = {
            "health_url": "http://localhost:11434",
            "health_method": "GET",
            "health_payload": None
        }
        assert check_health(server) is False


def test_check_health_returns_false_on_non_200():
    with patch("start.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500)
        server = {
            "health_url": "http://localhost:11434",
            "health_method": "GET",
            "health_payload": None
        }
        assert check_health(server) is False


# save_pids / load_pids 

def test_save_pids_writes_json():
    m = mock_open()
    with patch("builtins.open", m):
        save_pids({"Ollama": 1234, "Vector store MCP": 5678})
        written = "".join(call.args[0] for call in m().write.call_args_list)
        data = json.loads(written)
        assert data["Ollama"] == 1234
        assert data["Vector store MCP"] == 5678


def test_load_pids_returns_dict():
    fake_pids = {"Ollama": 1234}
    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(fake_pids))):
        result = load_pids()
        assert result["Ollama"] == 1234


def test_load_pids_returns_empty_when_no_file():
    with patch("pathlib.Path.exists", return_value=False):
        result = load_pids()
        assert result == {}


# start_server 

def test_start_server_skips_if_port_in_use():
    with patch("start.is_port_in_use", return_value=True):
        server = {
            "name": "Test server",
            "port": 8001,
            "cmd": ["uvicorn", "test:app"],
            "startup_delay": 0,
            "health_url": "http://localhost:8001",
            "health_method": "GET",
            "health_payload": None
        }
        result = start_server(server)
        assert result is None


def test_start_server_returns_pid_when_started():
    mock_proc = MagicMock()
    mock_proc.pid = 9999

    with patch("start.is_port_in_use", return_value=False), \
         patch("start.subprocess.Popen", return_value=mock_proc), \
         patch("start.time.sleep"), \
         patch("start.check_health", return_value=True):
        server = {
            "name": "Test server",
            "port": 8001,
            "cmd": ["uvicorn", "test:app"],
            "startup_delay": 0,
            "health_url": "http://localhost:8001",
            "health_method": "GET",
            "health_payload": None
        }
        result = start_server(server)
        assert result == 9999


def test_start_server_shows_pending_when_health_check_fails():
    mock_proc = MagicMock()
    mock_proc.pid = 9999

    with patch("start.is_port_in_use", return_value=False), \
         patch("start.subprocess.Popen", return_value=mock_proc), \
         patch("start.time.sleep"), \
         patch("start.check_health", return_value=False), \
         patch("builtins.print") as mock_print:
        server = {
            "name": "Test server",
            "port": 8001,
            "cmd": ["uvicorn", "test:app"],
            "startup_delay": 0,
            "health_url": "http://localhost:8001",
            "health_method": "GET",
            "health_payload": None
        }
        start_server(server)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "pending" in printed.lower()


# stop_servers 

def test_stop_servers_calls_taskkill_on_windows():
    fake_pids = {"Ollama": 1234, "Vector store MCP": 5678}

    with patch("start.load_pids", return_value=fake_pids), \
         patch("sys.platform", "win32"), \
         patch("start.subprocess.call") as mock_call, \
         patch("pathlib.Path.unlink"):
        stop_servers()
        assert mock_call.call_count == 2


def test_stop_servers_calls_sigterm_on_unix():
    fake_pids = {"Ollama": 1234}

    with patch("start.load_pids", return_value=fake_pids), \
         patch("sys.platform", "linux"), \
         patch("start.os.kill") as mock_kill, \
         patch("pathlib.Path.unlink"):
        stop_servers()
        mock_kill.assert_called_once()


def test_stop_servers_does_nothing_when_no_pids():
    with patch("start.load_pids", return_value={}), \
         patch("builtins.print") as mock_print:
        stop_servers()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "nothing" in printed.lower()


def test_stop_servers_handles_kill_failure_gracefully():
    fake_pids = {"Ollama": 1234}

    with patch("start.load_pids", return_value=fake_pids), \
         patch("sys.platform", "linux"), \
         patch("start.os.kill", side_effect=Exception("no such process")), \
         patch("pathlib.Path.unlink"):
        try:
            stop_servers()
        except Exception:
            assert False, "stop_servers should not raise"


def test_stop_servers_removes_pid_file():
    fake_pids = {"Ollama": 1234}

    with patch("start.load_pids", return_value=fake_pids), \
         patch("sys.platform", "linux"), \
         patch("start.os.kill"), \
         patch("pathlib.Path.unlink") as mock_unlink:
        stop_servers()
        mock_unlink.assert_called_once()


# SERVERS config 

def test_servers_config_has_all_required_servers():
    names = [s["name"] for s in SERVERS]
    assert "Ollama" in names
    assert "Vector store MCP" in names
    assert "Web search MCP" in names
    assert "arXiv MCP" in names
    assert "FastAPI main API" in names
    assert "Streamlit UI" in names
    assert "A2A agent server" in names


def test_servers_config_a2a_marked_correctly():
    a2a = next(s for s in SERVERS if s["name"] == "A2A agent server")
    assert a2a["a2a_only"] is True


def test_servers_config_non_a2a_servers_not_marked():
    non_a2a = [s for s in SERVERS if s["name"] != "A2A agent server"]
    assert all(not s["a2a_only"] for s in non_a2a)


def test_servers_config_all_have_required_keys():
    required = ["name", "cmd", "port", "health_url", "health_method", "startup_delay", "a2a_only"]
    for server in SERVERS:
        for key in required:
            assert key in server, f"{server['name']} missing key {key}"


def test_servers_config_ports_are_unique():
    ports = [s["port"] for s in SERVERS]
    assert len(ports) == len(set(ports))


# main 

def test_main_calls_stop_when_stop_flag():
    with patch("sys.argv", ["start.py", "--stop"]), \
         patch("start.stop_servers") as mock_stop:
        main()
        mock_stop.assert_called_once()


def test_main_skips_a2a_server_without_flag():
    started = []

    def fake_start(server):
        started.append(server["name"])
        return None

    with patch("sys.argv", ["start.py"]), \
         patch("start.start_server", side_effect=fake_start), \
         patch("start.save_pids"):
        main()
        assert "A2A agent server" not in started


def test_main_includes_a2a_server_with_flag():
    started = []

    def fake_start(server):
        started.append(server["name"])
        return 1234

    with patch("sys.argv", ["start.py", "--a2a"]), \
         patch("start.start_server", side_effect=fake_start), \
         patch("start.save_pids"):
        main()
        assert "A2A agent server" in started