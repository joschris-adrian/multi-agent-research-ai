# Multi-Agent Research Assistant

I built this to explore how multiple LLM agents can collaborate on a research task — each one handling a specific job rather than dumping everything into a single prompt.

The system runs fully locally using Ollama, so no API keys or costs involved.

---

## How it works

When you ask a question, five agents run in sequence:

1. **Planner** breaks the question into concrete research tasks
2. **Researcher** searches the web via DuckDuckGo and stores results in ChromaDB
3. **Analyst** pulls from both current results and past memory to extract insights
4. **Writer** turns those insights into a structured report
5. **Critic** reviews the report and flags anything missing or unclear

The vector memory means the system gets slightly smarter over repeated queries on similar topics — it can pull in relevant context from previous searches.

If DuckDuckGo rate-limits or returns nothing, the researcher retries up to 3 times with a short delay before giving up and returning an empty result set rather than crashing the pipeline.

---

## Stack

- **LLM:** Ollama (llama3.2) - runs locally, no API needed
- **Web search:** DuckDuckGo via `ddgs`
- **Vector memory:** ChromaDB
- **API:** FastAPI
- **UI:** Streamlit

---

## Project structure

```
multi-agent-research-ai/
├── main.py
├── conftest.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py     # Ollama API call + system prompt
│   │   ├── planner.py
│   │   ├── researcher.py     # DuckDuckGo search with retry + ChromaDB write
│   │   ├── analyst.py        # ChromaDB read + insight extraction
│   │   ├── writer.py
│   │   └── critic.py
│   ├── memory/
│   │   └── vector_store.py   # ChromaDB wrapper
│   ├── evaluation/
│   │   ├── evaluate.py       # Runs multi-agent vs single-agent comparison
│   │   ├── evaluator.py      # LLM-as-judge scoring
│   │   └── baseline.py       # Single prompt baseline
│   └── workflow/
│       └── agent_pipeline.py # Wires all agents together
│
├── api/
│   └── main.py               # FastAPI wrapper around the pipeline
├── ui/
│   └── streamlit_app.py      # Calls the API, renders results
└── tests/
    ├── test_agents.py
    ├── test_api.py
    ├── test_pipeline.py
    └── test_docker.py
```

---

## Running locally

You'll need [Ollama](https://ollama.com) installed.

```bash
ollama pull llama3.2
ollama serve
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Option 1 — CLI:**
```bash
python main.py
```

**Option 2 — API + UI (two terminals):**
```bash
# terminal 1
uvicorn api.main:app --reload

# terminal 2
streamlit run ui/streamlit_app.py
```

UI runs at `http://localhost:8501`, API docs at `http://127.0.0.1:8000/docs`.

**Option 3 — Docker:**
```bash
docker-compose up --build

# first time only — pull the model into the container
bash setup.sh
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests use mocks so Ollama doesn't need to be running.

---

## Evaluation

I ran the multi-agent pipeline against a single-agent baseline (same model, one prompt) on the query *"What are the latest trends in renewable energy?"* and scored both using an LLM judge.

| Criteria     | Multi-agent | Single-agent |
|--------------|-------------|--------------|
| Relevance    | 9/10        | 9/10         |
| Completeness | 8.5/10      | 8/10         |
| Clarity      | 9/10        | 9/10         |
| Accuracy     | 8/10        | 8.5/10       |

The multi-agent output was better structured and more complete. The single-agent scored slightly higher on accuracy — likely because it cited sources inline rather than summarising them. Overall the scores are close, which is expected given both use the same underlying model. The main benefit of the pipeline is the structured, readable output format.

---

## Known limitations

- `llama3.2` is a 3B model — outputs can be vague or repetitive on complex topics. Swapping to `mistral` or `llama3.1:8b` gives noticeably better results.
- ChromaDB runs in-memory by default, so vector memory resets on each restart. Switching to a persistent client fixes this.
- DuckDuckGo occasionally rate-limits — the researcher retries up to 3 times but will return an empty result if all attempts fail.

---

## Possible next steps

- Add source citations directly in the final report
- Persistent ChromaDB storage across sessions
- Streaming responses so you can watch the agents work in real time
- API authentication for deployment

---

## License

MIT
