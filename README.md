# Multi-Agent Research Assistant

I built this to explore how multiple LLM agents can collaborate on a research task - each one handling a specific job rather than dumping everything into a single prompt.

The system runs fully locally using Ollama, so no API keys or costs involved.

---

## How it works

When you ask a question, six agents run in sequence:

1. **Planner** breaks the question into concrete research tasks
2. **Researcher** calls the MCP web search server (DuckDuckGo) and stores results via the MCP vector store server (ChromaDB)
3. **Analyst** pulls from both current results and past memory via MCP vector store to extract insights
4. **Graph Builder** extracts entities (companies, trends, technologies) and stores relationships in Neo4j - runs at low temperature for consistent JSON output
5. **Writer** turns those insights into a structured report - runs at slightly higher temperature for more varied prose
6. **Critic** reviews the report and flags anything missing or unclear

The vector memory means the system gets slightly smarter over repeated queries on similar topics. The knowledge graph lets you query relationships between entities via GraphQL.

If the MCP web search server rate-limits or returns nothing, the researcher retries up to 3 times with a short delay before giving up and returning an empty result set rather than crashing the pipeline.

---

## Stack

| Component       | Technology                      |
|-----------------|---------------------------------|
| LLM             | Ollama (llama3.2)               |
| Agent pipeline  | Custom multi-agent architecture |
| Web search      | DuckDuckGo via `ddgs` + MCP     |
| Vector memory   | ChromaDB via MCP                |
| MCP servers     | FastAPI (vector store: 8001, web search: 8002) |
| Knowledge graph | Neo4j                           |
| GraphQL API     | Strawberry                      |
| Backend         | FastAPI                         |
| Frontend        | Streamlit                       |
| Fine-tuning     | PEFT (LoRA) via Hugging Face    |

---

## Project structure

```
multi-agent-research-ai/
├── main.py
├── run_all.py
├── conftest.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # Ollama API call + system prompt
│   │   ├── planner.py
│   │   ├── researcher.py       # Calls MCP web search server + MCP vector store
│   │   ├── analyst.py          # Calls MCP vector store for past memory + insight extraction
│   │   ├── graph_builder.py    # Extracts entities for Neo4j
│   │   ├── writer.py           # Ollama or fine-tuned LoRA model
│   │   └── critic.py
│   ├── memory/
│   │   └── vector_store.py     # ChromaDB wrapper
│   ├── mcp/
│   │   ├── client/
│   │   │   └── mcp_client.py          # HTTP client used by all agents
│   │   └── servers/
│   │       ├── vector_store_server.py # ChromaDB exposed as MCP server
│   │       └── web_search_server.py   # DuckDuckGo exposed as MCP server
│   ├── models/
│   │   └── peft_model.py       # LoRA adapter loader
│   ├── graph/
│   │   └── knowledge_graph.py  # Neo4j wrapper
│   ├── graphql/
│   │   └── graphql_schema.py   # Strawberry GraphQL schema
│   ├── evaluation/
│   │   ├── evaluate.py         # Multi-agent vs single-agent comparison
│   │   ├── evaluator.py        # LLM-as-judge scoring
│   │   └── baseline.py         # Single prompt baseline
│   └── workflow/
│       └── agent_pipeline.py   # Wires all agents together
│
├── training/
│   ├── generate_training_data.py  # Auto-generates examples using your agents
│   ├── dataset.json               # Training examples
│   ├── finetune.py                # LoRA fine-tuning script
│   └── evaluate_finetuning.py     # Before vs after comparison
│
├── models/
│   └── lora-adapter/           # Saved after running finetune.py
│
├── api/
│   └── main.py                 # FastAPI + GraphQL router
├── ui/
│   └── streamlit_app.py        # Calls the API, renders results
└── tests/
    ├── test_agents.py
    ├── test_api.py
    ├── test_pipeline.py
    ├── test_knowledge_graph.py
    ├── test_finetuning.py
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

**Option 1 - CLI:**
```bash
python main.py
```

**Option 2 - API + UI (four terminals):**
```bash
# terminal 1
uvicorn api.main:app --reload

# terminal 2
uvicorn src.mcp.servers.vector_store_server:app --port 8001 --reload

# terminal 3
uvicorn src.mcp.servers.web_search_server:app --port 8002 --reload

# terminal 4
streamlit run ui/streamlit_app.py
```

UI at `http://localhost:8501`, API docs at `http://127.0.0.1:8000/docs`, GraphQL at `http://127.0.0.1:8000/graphql`.

**Option 3 - Docker (includes Neo4j):**
```bash
docker-compose up --build
bash setup.sh  # first time only
```

---

## Fine-tuning (PEFT + LoRA)

The writer agent can be swapped between the default Ollama model and a locally fine-tuned LoRA adapter.

```bash
# step 1 - generate training data using your own agents (ollama must be running)
python training/generate_training_data.py

# step 2 - fine-tune (opt-125m, runs on CPU in ~2 min)
python training/finetune.py

# step 3 - run with fine-tuned writer
set USE_FINETUNED=1 && python main.py   # Windows
USE_FINETUNED=1 python main.py          # Linux/Mac

# step 4 - evaluate before vs after
python training/evaluate_finetuning.py
```

The writer falls back to Ollama automatically if the adapter hasn't been trained or if `USE_FINETUNED` is not set.

---

## Running tests

```bash
# fast run - excludes slow torch/finetuning tests (~2 min)
pytest tests/ -m "not finetuning" -v

# full run including finetuning tests (~10 min)
pytest tests/ -v

# parallel run (install pytest-xdist first)
# note: use -n 2, not -n auto — higher concurrency causes OOM crashes
# when multiple workers load the ONNX embedding model simultaneously
pip install pytest-xdist
pytest tests/ -m "not finetuning" -n 2
```

Tests use mocks so Ollama and Neo4j don't need to be running.

To verify all components end to end (requires Ollama, uvicorn, and both MCP servers running):

```bash
# terminal 1
ollama serve

# terminal 2
uvicorn api.main:app --reload

# terminal 3
uvicorn src.mcp.servers.vector_store_server:app --port 8001 --reload

# terminal 4
uvicorn src.mcp.servers.web_search_server:app --port 8002 --reload

# terminal 5
python run_all.py
```

Neo4j checks are skipped automatically if Docker isn't running. To include them:

```bash
docker start neo4j
```

---

## Evaluation

### Multi-agent vs single-agent

I ran the pipeline against a single-agent baseline on the query *"What are the latest trends in renewable energy?"* and scored both using an LLM judge.

| Criteria     | Multi-agent | Single-agent |
|--------------|-------------|--------------|
| Relevance    | 9/10        | 9/10         |
| Completeness | 8.5/10      | 8/10         |
| Clarity      | 9/10        | 9/10         |
| Accuracy     | 8/10        | 8.5/10       |

The multi-agent output was better structured and more complete. The single-agent scored slightly higher on accuracy due to citing sources inline.

### Fine-tuning evaluation

I evaluated the LoRA fine-tuned writer (opt-125m, 125M params, 10 examples) against the Ollama baseline (llama3.2, 3B params).

| Criteria     | Ollama (llama3.2) | LoRA (opt-125m) |
|--------------|-------------------|-----------------|
| Relevance    | 9/10              | 4/10            |
| Completeness | 8/10              | 6/10            |
| Clarity      | 9/10              | 7/10            |
| Accuracy     | 8/10              | 5/10            |

The fine-tuned model underperforms because opt-125m is 24x smaller than llama3.2 and was trained on only 10 examples. The purpose of this component is to demonstrate the end-to-end fine-tuning workflow - data generation, LoRA training, adapter loading, and quantitative evaluation - rather than to beat a much larger model. A fair comparison would require fine-tuning a model of comparable size with significantly more training data.

---

## Known limitations

- `llama3.2` is a 3B model - outputs can be vague on complex topics. `mistral` or `llama3.1:8b` give better results.
- ChromaDB runs in-memory by default, so vector memory resets on each restart.
- ChromaDB's ONNX embedding model takes 20-30 seconds to initialise on first write. The MCP vector store client uses a 60 second timeout to handle this - subsequent calls are fast.
- DuckDuckGo occasionally rate-limits - the MCP web search server retries 3 times before returning empty.
- MCP servers must be running separately for agents to access web search and vector memory. Vector store runs on port 8001, web search on port 8002. The analyst degrades gracefully if unavailable, but the researcher will return empty results.
- The graph builder relies on the LLM returning valid JSON - falls back to empty entity set if parsing fails.
- Neo4j must be running separately (via Docker) for the knowledge graph to work. The pipeline skips it gracefully if unavailable.
- The LoRA fine-tuned writer uses opt-125m which is too small for high-quality report generation without significantly more training data.

---

## Possible next steps

- Fine-tune a larger model (e.g. llama3.2) with more training data for meaningful quality improvement
- Add source citations directly in the final report
- Persistent ChromaDB storage across sessions
- Streaming responses via WebSockets
- Visualise the knowledge graph in the Streamlit UI
- API authentication for deployment
- Add an MCP server for Neo4j to fully decouple the knowledge graph from agents

---

## License

MIT
