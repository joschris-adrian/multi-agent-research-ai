"""
End-to-end workflow test.
Runs every component of the system in sequence and reports pass/fail.

Usage:
    python run_all.py

Requirements:
    - Ollama running with llama3.2 pulled
    - uvicorn api.main:app --reload running in another terminal
    - Docker optional (Neo4j skipped if unavailable)
"""

import sys
import os
import subprocess
import requests

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

results = []

NEO4J_ERROR = "7687"


def is_neo4j_error(e):
    return NEO4J_ERROR in str(e)


def check(name, fn):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    try:
        fn()
        print(f"  PASS: {name}")
        results.append((name, "PASS", ""))
    except Exception as e:
        if is_neo4j_error(e):
            print("  SKIP: Neo4j not running — start with Docker to test")
            results.append((name, "SKIP", "Neo4j not running"))
        else:
            print(f"  FAIL: {name}")
            print(f"  {e}")
            results.append((name, "FAIL", str(e)))


def check_tests():
    result = subprocess.run(
        ["pytest", "tests/", "-m", "not finetuning", "--tb=no", "-q"],
        capture_output=True, text=True
    )
    print(result.stdout[-800:])
    assert result.returncode == 0, "some tests failed"

check("Fast pytest suite (excluding finetuning)", check_tests)


def check_ollama():
    r = requests.get("http://localhost:11434", timeout=5)
    assert r.status_code == 200, f"ollama returned {r.status_code}"
    print("  ollama is running")

check("Ollama health", check_ollama)


def check_single_agent():
    from src.agents.base_agent import BaseAgent
    agent = BaseAgent(role="Tester", goal="Return short answers")
    result = agent.run("Say OK in one word.")
    assert isinstance(result, str) and len(result) > 0
    print(f"  agent response: {result[:60]}")

check("Single agent (BaseAgent)", check_single_agent)


def check_planner():
    from src.agents.planner import PlannerAgent
    tasks = PlannerAgent().plan("What are trends in solar energy?")
    assert isinstance(tasks, str) and len(tasks) > 0
    print(f"  planner output: {tasks[:80]}...")

check("Planner agent", check_planner)


def check_researcher():
    from src.agents.researcher import ResearchAgent
    agent = ResearchAgent()
    docs = agent.search("solar energy trends 2025", max_results=2)
    assert isinstance(docs, list)
    print(f"  found {len(docs)} documents")
    if docs:
        print(f"  first result: {docs[0]['title'][:60]}")

check("Researcher + ChromaDB", check_researcher)


def check_analyst():
    from src.agents.analyst import AnalystAgent
    docs = [{"title": "Solar boom", "content": "Solar energy grew 40% in 2024.", "source": "http://example.com"}]
    insights = AnalystAgent().analyze(docs, "solar energy trends")
    assert isinstance(insights, str) and len(insights) > 0
    print(f"  insights: {insights[:80]}...")

check("Analyst agent", check_analyst)


def check_writer():
    from src.agents.writer import WriterAgent
    entities = {"companies": ["Tesla"], "trends": ["Solar growth"], "technologies": ["Battery storage"]}
    report = WriterAgent().write_report("Solar energy is growing rapidly.", entities)
    assert isinstance(report, str) and len(report) > 0
    print(f"  report preview: {report[:80]}...")

check("Writer agent (with entities)", check_writer)


def check_critic():
    from src.agents.critic import CriticAgent
    feedback = CriticAgent().review("Solar energy is growing. The end.")
    assert isinstance(feedback, str) and len(feedback) > 0
    print(f"  feedback: {feedback[:80]}...")

check("Critic agent", check_critic)


def check_pipeline():
    from src.workflow.agent_pipeline import MultiAgentResearchSystem
    system = MultiAgentResearchSystem()
    result = system.run("What are the latest trends in solar energy?")
    for key in ["question", "tasks", "documents", "insights", "entities", "report", "critic_feedback"]:
        assert key in result, f"missing key: {key}"
    assert len(result["report"]) > 50
    print(f"  report length: {len(result['report'])} chars")
    print(f"  entities found: {list(result['entities'].keys())}")
    print(f"  companies: {result['entities'].get('companies', [])}")

check("Full pipeline (all agents)", check_pipeline)


def check_neo4j():
    from src.graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_topic("solar energy test")
    kg.add_entity("Tesla", "Company")
    kg.link_entity_to_topic("Tesla", "solar energy test")
    entities = kg.get_entities_for_topic("solar energy test")
    assert any(e["name"] == "Tesla" for e in entities)
    kg.close()
    print(f"  found {len(entities)} entities for test topic")

check("Neo4j knowledge graph", check_neo4j)


def check_graph_builder():
    from src.agents.graph_builder import GraphBuilderAgent
    agent = GraphBuilderAgent()
    entities = agent.extract_entities(
        "Tesla dominates EV market. Battery storage is a key trend.",
        "electric vehicles"
    )
    assert isinstance(entities, dict)
    assert "companies" in entities
    print(f"  companies: {entities.get('companies', [])}")
    print(f"  trends: {entities.get('trends', [])}")

check("Graph builder agent", check_graph_builder)


def check_api():
    try:
        r = requests.get("http://localhost:8000", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "message" in data
        print(f"  root endpoint: {data['message']}")

        r2 = requests.post("http://localhost:8000/research", json={}, timeout=10)
        assert r2.status_code == 422
        print("  /research endpoint exists and validates schema correctly")

    except requests.exceptions.ConnectionError:
        raise Exception(
            "FastAPI not running. Start it with: uvicorn api.main:app --reload"
        )

check("FastAPI endpoints", check_api)


def check_graphql():
    try:
        r = requests.post(
            "http://localhost:8000/graphql",
            json={"query": "{ entities { name kind } }"},
            timeout=10
        )
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        entities = (data.get("data") or {}).get("entities") or []
        print(f"  GraphQL endpoint reachable, returned {len(entities)} entities")
    except requests.exceptions.ConnectionError:
        raise Exception(
            "FastAPI not running. Start it with: uvicorn api.main:app --reload"
        )

check("GraphQL endpoint", check_graphql)


def check_lora():
    assert os.path.exists("models/lora-adapter"), \
        "adapter not found — run: python training/finetune.py"
    files = os.listdir("models/lora-adapter")
    assert any("adapter" in f for f in files), \
        f"adapter files missing, got: {files}"
    print(f"  adapter files: {[f for f in files if 'adapter' in f]}")

check("LoRA adapter (file check)", check_lora)


print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")

passed = [r for r in results if r[1] == "PASS"]
skipped = [r for r in results if r[1] == "SKIP"]
failed = [r for r in results if r[1] == "FAIL"]

for name, status, _ in results:
    icon = "✓" if status == "PASS" else ("~" if status == "SKIP" else "✗")
    print(f"  {icon}  {name}")

print(f"\n  {len(passed)}/{len(results) - len(skipped)} checks passed", end="")
if skipped:
    print(f"  ({len(skipped)} skipped — Neo4j not running)", end="")
print()

if failed:
    print("\n  Failed checks:")
    for name, _, reason in failed:
        print(f"    - {name}: {reason[:100]}")
    sys.exit(1)
else:
    print("\n  All checks passed.")
