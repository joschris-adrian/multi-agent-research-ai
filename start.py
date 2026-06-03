"""
Start all servers required for the multi-agent research system.

Usage:
    python start.py          # starts all servers
    python start.py --a2a    # starts all servers including A2A
    python start.py --stop   # stops all running servers

Servers started:
    - Ollama                        (port 11434)
    - Vector store MCP              (port 8001)
    - Web search MCP                (port 8002)
    - arXiv MCP                     (port 8003)
    - Vector store MCP protocol     (port 8011)
    - Web search MCP protocol       (port 8012)
    - arXiv MCP protocol            (port 8013)
    - FastAPI main API              (port 8000)
    - Streamlit UI                  (port 8501)
    - A2A agent server              (port 8004, optional)
"""

import argparse
import subprocess
import sys
import os
import time
import requests
import json
from pathlib import Path

PID_FILE = ".server_pids.json"

SERVERS = [
    {
        "name": "Ollama",
        "cmd": ["ollama", "serve"],
        "port": 11434,
        "health_url": "http://localhost:11434",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 3,
        "a2a_only": False,
    },
    {
        "name": "Vector store MCP",
        "cmd": ["uvicorn", "src.mcp.servers.vector_store_server:app", "--port", "8001"],
        "port": 8001,
        "health_url": "http://localhost:8001/vector_store/search",
        "health_method": "POST",
        "health_payload": {"query": "health"},
        "startup_delay": 2,
        "a2a_only": False,
    },
    {
        "name": "Web search MCP",
        "cmd": ["uvicorn", "src.mcp.servers.web_search_server:app", "--port", "8002"],
        "port": 8002,
        "health_url": "http://localhost:8002/web_search/search",
        "health_method": "POST",
        "health_payload": {"query": "health", "max_results": 1, "retries": 1, "delay": 0},
        "startup_delay": 2,
        "a2a_only": False,
    },
    {
        "name": "arXiv MCP",
        "cmd": ["uvicorn", "src.mcp.servers.arxiv_server:app", "--port", "8003"],
        "port": 8003,
        "health_url": "http://localhost:8003/arxiv/search",
        "health_method": "POST",
        "health_payload": {"topic": "health", "max_results": 1},
        "startup_delay": 2,
        "a2a_only": False,
    },
    {
        "name": "Vector store MCP (protocol)",
        "cmd": ["python", "-m", "src.mcp.servers.vector_store_mcp_server"],
        "port": 8011,
        "health_url": "http://localhost:8011/mcp",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 5,
        "a2a_only": False,
    },
    {
        "name": "Web search MCP (protocol)",
        "cmd": ["python", "-m", "src.mcp.servers.web_search_mcp_server"],
        "port": 8012,
        "health_url": "http://localhost:8012/mcp",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 3,
        "a2a_only": False,
    },
    {
        "name": "arXiv MCP (protocol)",
        "cmd": ["python", "-m", "src.mcp.servers.arxiv_mcp_server"],
        "port": 8013,
        "health_url": "http://localhost:8013/mcp",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 3,
        "a2a_only": False,
    },
    {
        "name": "FastAPI main API",
        "cmd": ["uvicorn", "api.main:app", "--port", "8000"],
        "port": 8000,
        "health_url": "http://localhost:8000",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 2,
        "a2a_only": False,
    },
    {
        "name": "Streamlit UI",
        "cmd": ["streamlit", "run", "ui/streamlit_app.py"],
        "port": 8501,
        "health_url": "http://localhost:8501",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 5,
        "a2a_only": False,
    },
    {
        "name": "A2A agent server",
        "cmd": ["uvicorn", "src.a2a.agent_server:app", "--port", "8004"],
        "port": 8004,
        "health_url": "http://localhost:8004/health",
        "health_method": "GET",
        "health_payload": None,
        "startup_delay": 2,
        "a2a_only": True,
    },
]


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def check_health(server, timeout=60):
    url = server["health_url"]
    method = server["health_method"]
    payload = server["health_payload"]
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code in (200, 406)
    except Exception:
        return False
    

def start_server(server):
    if is_port_in_use(server["port"]):
        print(f"  ✓ {server['name']} already running on port {server['port']}")
        return None

    print(f"  starting {server['name']} on port {server['port']}...")
    proc = subprocess.Popen(
        server["cmd"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    time.sleep(server["startup_delay"])

    if check_health(server, timeout=10):
        print(f"  ✓ {server['name']} ready")
    else:
        print(f"  ~ {server['name']} started (health check pending)")

    return proc.pid


def save_pids(pids):
    with open(PID_FILE, "w") as f:
        json.dump(pids, f)


def load_pids():
    if not Path(PID_FILE).exists():
        return {}
    with open(PID_FILE) as f:
        return json.load(f)


def stop_servers():
    pids = load_pids()
    if not pids:
        print("No servers tracked. Nothing to stop.")
        return

    import signal
    for name, pid in pids.items():
        try:
            if sys.platform == "win32":
                subprocess.call(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"  ✓ stopped {name} (pid {pid})")
        except Exception as e:
            print(f"  ✗ could not stop {name} (pid {pid}): {e}")

    Path(PID_FILE).unlink(missing_ok=True)
    print("\nAll servers stopped.")


def main():
    parser = argparse.ArgumentParser(description="Start all research system servers")
    parser.add_argument("--a2a", action="store_true", help="Also start the A2A agent server")
    parser.add_argument("--stop", action="store_true", help="Stop all running servers")
    args = parser.parse_args()

    if args.stop:
        stop_servers()
        return

    print("\nStarting research system servers...")
    print("=" * 50)

    pids = {}
    for server in SERVERS:
        if server["a2a_only"] and not args.a2a:
            continue
        pid = start_server(server)
        if pid:
            pids[server["name"]] = pid

    if pids:
        save_pids(pids)

    print("\n" + "=" * 50)
    print("  All servers started.")
    print(f"  UI:      http://localhost:8501")
    print(f"  API:     http://localhost:8000/docs")
    print(f"  GraphQL: http://localhost:8000/graphql")
    if args.a2a:
        print(f"  A2A:     http://localhost:8004/health")
    print("\n  To stop all servers: python start.py --stop")
    print("  To run pipeline:     python main.py")
    if args.a2a:
        print("  To run A2A pipeline: python run_a2a.py")
    print()


if __name__ == "__main__":
    main()