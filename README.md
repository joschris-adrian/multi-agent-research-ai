# Multi-Agent Research Assistant

I built this to explore how multiple LLM agents can collaborate on a research task - each one handling a specific job rather than dumping everything into a single prompt.

The system runs locally using Ollama by default, with optional Google Gemini support via the free tier on Google AI Studio.

---

## How it works

When you ask a question, six agents run in sequence:

1. **Planner** breaks the question into concrete research tasks
2. **Researcher** calls both the MCP web search server (DuckDuckGo) and the MCP arXiv server in parallel, merges results, chunks them into 200-character overlapping segments and stores them via the MCP vector store server (ChromaDB). The arXiv server retrieves the most recent academic papers sorted by submission date.
3. **Analyst** retrieves semantically ranked chunks from the MCP vector store (RAG), filters by relevance score and injects the top results into the prompt alongside current research
4. **Graph Builder** extracts entities (companies, trends, technologies) and stores relationships in Neo4j - runs at low temperature for consistent JSON output
5. **Writer** turns those insights into a structured report - runs at slightly higher temperature for more varied prose
6. **Critic** reviews the report and flags anything missing or unclear

An alternative A2A pipeline (`src/a2a/a2a_pipeline.py`) exposes each agent as an independent HTTP endpoint on port 8004. Unlike the fixed sequential pipeline, the A2A coordinator can dynamically re-invoke agents - requesting additional research if analyst insights are insufficient, or asking the writer to revise the report if the critic flags major issues.

Before running the pipeline on a topic, you can optionally pre-populate vector memory using supercharge mode - this gives the analyst richer past context to retrieve from on the first run.

The vector memory means the system gets slightly smarter over repeated queries on similar topics. The knowledge graph lets you query relationships between entities via GraphQL.

If the MCP web search server rate-limits or returns nothing, the researcher retries up to 3 times with a short delay before giving up and returning an empty result set rather than crashing the pipeline.

---

## Stack

| Component       | Technology                      |
|-----------------|---------------------------------|
| LLM             | Ollama (llama3.2) or Google Gemini (gemini-2.5-flash) |
| Agent pipeline  | Custom multi-agent architecture |
| Web search      | DuckDuckGo via `ddgs` + MCP     |
| Vector memory   | ChromaDB via MCP                |
| MCP servers (HTTP) | FastAPI (vector store: 8001, web search: 8002, arXiv: 8003) |
| MCP servers (protocol) | FastMCP streamable HTTP (vector store: 8011, web search: 8012, arXiv: 8013) |
| MCP client      | Official Python MCP SDK (streamablehttp_client)          |
| A2A             | Agent-to-Agent protocol (port 8004)             |
| Supercharge     | `scripts/supercharge.py` - bulk ingestion CLI   |
| RAG             | ChromaDB semantic search + cross-encoder reranker (ms-marco-MiniLM-L-6-v2) |
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
├── scripts/
│   ├── supercharge.py          # Bulk ingest documents into ChromaDB without running pipeline
│   └── run_subscriptions.py    # CLI trigger for the subscription scheduler
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
│   ├── a2a/
│   │   ├── agent_server.py     # Exposes each agent as FastAPI endpoint
│   │   ├── a2a_client.py       # HTTP client for agent-to-agent calls
│   │   └── a2a_pipeline.py     # Dynamic pipeline with re-research and re-write
│   ├── memory/
│   │   └── vector_store.py     # ChromaDB wrapper
│   ├── mcp/
│   │   ├── client/
│   │   └── mcp_client.py          # MCP SDK client used by all agents
│   │   └── servers/
│   │       ├── vector_store_server.py # ChromaDB exposed as MCP server
│   │       ├── web_search_server.py   # DuckDuckGo exposed as MCP server
│   │       ├── arxiv_server.py        # arXiv API exposed as MCP server
│   │       ├── vector_store_mcp_server.py # ChromaDB as standalone FastMCP server (port 8011)
│   │       ├── web_search_mcp_server.py   # DuckDuckGo as standalone FastMCP server (port 8012)
│   │       └── arxiv_mcp_server.py        # arXiv as standalone FastMCP server (port 8013)
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
│   ├── scheduler/
│   │   ├── scheduler.py            # Checks due subscriptions and runs the pipeline
│   │   ├── subscription_store.py   # JSON-backed subscription persistence
│   │   └── delivery.py             # Email, Slack and Discord delivery
│   └── workflow/
│       └── agent_pipeline.py       # Wires all agents together
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

**Option 2a - Single command startup:**
```bash
# start all servers (Ollama, MCP servers, FastAPI, Streamlit)
python start.py

# start with Gemini for analyst, writer and critic (Ollama still required for planner, researcher, graph builder)
python start.py --gemini

# start including A2A pipeline server
python start.py --a2a

# start with Gemini and A2A
python start.py --gemini --a2a

# stop all servers started by start.py
python start.py --stop
```
Servers are started as background processes with health checks. PIDs are tracked in `.server_pids.json` and cleaned up on stop. If `LLM_PROVIDER=gemini` is already set in `.env`, `python start.py` activates Gemini mode automatically without needing the `--gemini` flag. Ollama is always started regardless of provider.


**Option 2b - API + UI (eight terminals):**
```bash
# terminal 1
uvicorn api.main:app --reload

# terminal 2
uvicorn src.mcp.servers.vector_store_server:app --port 8001 --reload

# terminal 3
uvicorn src.mcp.servers.web_search_server:app --port 8002 --reload

# terminal 4
uvicorn src.mcp.servers.arxiv_server:app --port 8003 --reload

# terminal 5
python -m src.mcp.servers.vector_store_mcp_server

# terminal 6
python -m src.mcp.servers.web_search_mcp_server

# terminal 7
python -m src.mcp.servers.arxiv_mcp_server

# terminal 8
streamlit run ui/streamlit_app.py
```

UI at `http://localhost:8501`, API docs at `http://127.0.0.1:8000/docs`, GraphQL at `http://127.0.0.1:8000/graphql`.

**Option 3 - Docker (includes Neo4j):**
ChromaDB data is persisted to a Docker volume (`chroma_data`) so vector memory survives container restarts.
```bash
docker-compose up --build
bash setup.sh  # first time only
```

**Option 4 - A2A pipeline CLI:**
```bash
# start A2A agent server in addition to other terminals
uvicorn src.a2a.agent_server:app --port 8004 --reload

# interactive
python run_a2a.py

# direct
python run_a2a.py --question "What are the latest trends in AI Agents?"
```
The A2A pipeline dynamically re-invokes agents when insights are insufficient or the critic flags issues, unlike the fixed sequential pipeline.

**Supercharge mode - pre-populate vector memory before running the pipeline:**
```bash
# requires MCP servers running on ports 8001 and 8002
python scripts/supercharge.py --topic "renewable energy" --max_results 5
```
This fetches documents across multiple query variations of the topic, chunks them and stores them directly in ChromaDB via the MCP vector store server. Run this before the main pipeline to give the analyst richer context from the first query. The supercharge script also queries the arXiv MCP server for academic papers on the topic alongside web search results. Requires the arXiv server running on port 8003.

**To run with A2A pipeline (additional terminal):**
```bash
uvicorn src.a2a.agent_server:app --port 8004 --reload
```
Then use `A2AResearchSystem` instead of `MultiAgentResearchSystem` in `main.py`.

---

## Using Gemini instead of Ollama

By default the pipeline runs locally via Ollama. You can switch to Google Gemini
instead — the free tier on Google AI Studio is sufficient for normal use.

### Get a Gemini API key

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key

### Configure the project

Open `.env` and set:
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here

Leave `OLLAMA_HOST` and `OLLAMA_MODEL` as they are — they are ignored when
`LLM_PROVIDER=gemini`.

### Test your connection

```bash
python scripts/test_gemini_connection.py
```

Expected output:
```
Testing Gemini connection...
Model: gemini-2.5-flash
Key:   AIzaSyAB********
Response: OK
Connection successful.
```

### Model assignment

| Agent | Gemini model | Reason |
|---|---|---|
| Analyst | gemini-2.5-flash | Higher quality insight extraction |
| Writer | gemini-2.5-flash | Better prose generation |
| Critic | gemini-2.5-flash | More precise feedback |

### Free tier limits

The free tier allows 15 requests per minute. Only the analyst, writer and critic
call Gemini — the planner, researcher and graph builder always use Ollama
regardless of `LLM_PROVIDER`, as model quality makes no measurable difference
for task breakdown, query extraction and JSON entity extraction. This means
Ollama must still be running even when `LLM_PROVIDER=gemini`. The pipeline
automatically adds a 10-second delay between the three Gemini agent calls to
stay within the rate limit. You can tune this via `GEMINI_DELAY_SECONDS` in
your `.env` if needed.

If you hit rate limit errors despite the delay, increase the delay by setting
`GEMINI_DELAY_SECONDS` in your `.env`:

GEMINI_DELAY_SECONDS=15


### Note on Ollama

Even when `LLM_PROVIDER=gemini`, Ollama must still be running — the planner,
researcher and graph builder always use Ollama regardless of provider. Only the
analyst, writer and critic call Gemini.

### Switching back to Ollama

Set `LLM_PROVIDER=ollama` in `.env` (or remove the line entirely). No other
changes are needed.

## Research topic subscriptions

Subscribe to a topic and receive a freshly generated report on a configurable
schedule via email, Slack, or Discord.

### Manage subscriptions via API

```bash
# subscribe to a topic
curl -X POST http://localhost:8000/subscriptions \
  -H "Content-Type: application/json" \
  -d '{"topic": "renewable energy", "frequency": "weekly", "delivery_method": "email", "delivery_target": "you@example.com"}'

# list subscriptions
curl http://localhost:8000/subscriptions

# pause a subscription
curl -X POST http://localhost:8000/subscriptions/{id}/pause

# resume a subscription
curl -X POST http://localhost:8000/subscriptions/{id}/resume

# delete a subscription
curl -X DELETE http://localhost:8000/subscriptions/{id}

# trigger the scheduler manually
curl -X POST http://localhost:8000/subscriptions/run
```

Or run the scheduler directly from the CLI:
```bash
python scripts/run_subscriptions.py
```

The CLI script checks all subscriptions and runs any that are due. It is safe to run repeatedly — subscriptions that are not yet due are skipped silently. Use this for manual testing or for wiring into a cron job or Windows Task Scheduler rather than relying on the API trigger.

To test the full subscription flow end to end without waiting for a scheduled interval, create a subscription with `"frequency": "daily"` and `"delivery_method": "log"`, then immediately run:

```bash
python scripts/run_subscriptions.py
```

The report will be printed to stdout. Switch `delivery_method` to `"email"`, `"slack"`, or `"discord"` and set the appropriate credentials in `.env` when ready to use live delivery.

### Email delivery setup (Gmail)

1. Enable 2-Step Verification at https://myaccount.google.com/security
2. Generate an app password at https://myaccount.google.com/apppasswords
3. Add to `.env`:

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_gmail_address@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx

Other providers (Outlook, Yahoo, SendGrid) work the same way — see their
SMTP settings and use port 587 with your credentials or API key.

### Slack and Discord delivery

Pass a webhook URL as `delivery_target`:
```bash
# Slack — create a webhook at https://api.slack.com/messaging/webhooks
# Discord — Server Settings > Integrations > Webhooks > New Webhook

curl -X POST http://localhost:8000/subscriptions \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI agents", "frequency": "weekly", "delivery_method": "slack", "delivery_target": "https://hooks.slack.com/services/..."}'
```

### Scheduling automatically

To run the scheduler on a fixed schedule without manual triggers, add a cron
job (Linux/Mac) or Task Scheduler entry (Windows):

```bash
# Linux/Mac — run every day at 8am
0 8 * * * cd /path/to/project && python scripts/run_subscriptions.py

# Windows Task Scheduler — action: program
python scripts/run_subscriptions.py
```

The scheduler only runs subscriptions that are due based on their configured
frequency — running it daily is safe even for weekly subscriptions.

## Fine-tuning using PEFT(LoRA)

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

To verify all components end to end (requires Ollama, uvicorn, all three MCP servers and the A2A server running):

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
uvicorn src.mcp.servers.arxiv_server:app --port 8003 --reload

# terminal 6
uvicorn src.a2a.agent_server:app --port 8004 --reload

# terminal 7
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
| Relevance    | 9/10        | 8/10         |
| Completeness | 8/10        | 9/10         |
| Clarity      | 8/10        | 9/10         |
| Accuracy     | 9/10        | 8.5/10       |

The multi-agent system produced a data-grounded report with real statistics retrieved from the web (585 GW of new capacity, 92.5% share of new electricity), structured using the enforced output schema. The single-agent produced a broader and more readable overview but drew entirely from training data with a December 2023 cutoff, with no live retrieval. The multi-agent scored higher on accuracy due to current retrieved data; the single-agent scored higher on completeness and clarity due to broader topic coverage and more natural prose. The enforced report schema in the writer constrained the multi-agent output to specific sections which limited narrative flow but improved consistency.

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

## Prompt engineering

Each agent uses a tailored prompt strategy suited to its role in the pipeline:

| Agent | Technique | Purpose |
|---|---|---|
| Planner | Few-shot example | Shows the model the expected task list format |
| Analyst | Chain-of-thought | Forces step-by-step reasoning before extracting insights |
| Writer | Output schema constraints | Enforces consistent report structure across runs |
| Critic | Adversarial persona | Produces sharper, more actionable feedback |
| Graph Builder | Negative prompting | Prevents hallucinated entities and generic terms in JSON |
| All agents | Role-specific system prompt | Each agent is aware it is part of a multi-agent pipeline and should not guess when information is unavailable |

## Reranker

The system uses a two-stage retrieval pipeline for more precise RAG:

**Stage 1 — Bi-encoder retrieval (ChromaDB)**
ChromaDB uses the ONNX embedding model to convert the query and all stored chunks into dense vectors and retrieves the top candidates by cosine similarity. This is fast but imprecise — cosine similarity measures general semantic closeness, not whether a chunk actually answers the query.

**Stage 2 — Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)**
The retrieved candidates are passed to a cross-encoder reranker which scores each (query, chunk) pair together rather than independently. This is slower but significantly more precise — the cross-encoder reads both the query and the chunk simultaneously and scores how well the chunk answers the specific query, not just how similar they are in embedding space.

**Why this is better:**
A bi-encoder might rank a chunk highly because it contains many of the same words as the query, even if it doesn't answer the question. The cross-encoder re-orders the candidates by actual relevance to the query, so the analyst receives the most directly useful chunks rather than just the most similar ones.

**Example:**
For the query "what is the market share of solar energy in 2024?", cosine similarity might rank a general chunk about solar energy trends highly because it shares many tokens. The reranker would demote that chunk and promote a specific chunk containing "solar energy accounted for 36% of new power capacity in 2024" because it directly answers the question.

**Fallback:**
If the reranker model is unavailable (no internet on first run, or insufficient memory), the system falls back to cosine similarity automatically with no change to the pipeline.

**Debug endpoint:**
To compare cosine-only vs reranked results on any query:
```bash
curl -X POST http://localhost:8001/vector_store/search/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "solar energy market share 2024", "top_k": 5}'
```
This returns both orderings side by side showing rerank scores vs cosine scores.

## Streaming (Server-Sent Events)

The `/research/stream` endpoint streams real-time progress updates from each agent as they complete, rather than blocking until the full pipeline finishes.

**How it works:**
The FastAPI endpoint runs each agent sequentially using `run_in_executor` to avoid blocking the async event loop. After each agent completes it yields a JSON SSE event with the agent name, status and a preview of the output. The Streamlit UI consumes the stream line by line and updates the progress bar and status box in real time.

**Event format:**
Each event is a JSON object on a `data:` line:
```json
{"event": "agent", "data": {"agent": "planner", "status": "running", "message": "Breaking question into tasks..."}}
{"event": "agent", "data": {"agent": "planner", "status": "done", "output": "1. Search trends..."}}
{"event": "complete", "data": {"question": "...", "report": "...", "entities": {...}, ...}}
```

**Event types:**
- `start` — pipeline has begun
- `agent` — an agent has started (`status: running`) or finished (`status: done`)
- `complete` — all agents done, full result payload included

**Why SSE over WebSockets:**
SSE is simpler for one-way server-to-client streaming — no handshake, no connection upgrade, works over standard HTTP. WebSockets would be needed for bidirectional communication (e.g. letting the user send a follow-up question mid-pipeline) which is a future improvement.

**Known limitation:**
The `/research` non-streaming endpoint remains available for direct API use and testing. The Streamlit UI uses `/research/stream` by default.

## Known limitations

- `llama3.2` is a 3B model - outputs can be vague on complex topics. `mistral` or `llama3.1:8b` give better results.
- ChromaDB persists to `./chroma_db` on disk. Chunks older than 7 days are automatically evicted on each search. Delete this folder to reset vector memory entirely. TTL is configurable via `TTL_SECONDS` in `vector_store.py`.
- ChromaDB's ONNX embedding model takes 20-30 seconds to initialise on first write. The MCP vector store client uses a 60 second timeout to handle this - subsequent calls are fast.
- DuckDuckGo occasionally rate-limits - the MCP web search server retries 3 times before returning empty.
- Two sets of MCP servers must be running separately. The HTTP API servers (ports 8001, 8002, 8003) handle custom endpoints used by health checks and the compare endpoint. The protocol servers (ports 8011, 8012, 8013) handle MCP-compliant tool calls from agents via the streamable HTTP transport. If the protocol servers are unavailable the researcher and analyst will return empty results.
- The graph builder relies on the LLM returning valid JSON - falls back to empty entity set if parsing fails.
- Neo4j must be running separately (via Docker) for the knowledge graph to work. The pipeline skips it gracefully if unavailable.
- The LoRA fine-tuned writer uses opt-125m which is too small for high-quality report generation without significantly more training data.
- Web search results are chunked into 200-character segments with 50-character overlap before storing in ChromaDB. DuckDuckGo snippets are short so each result typically produces 2-3 chunks.
- ChromaDB relevance scores use cosine distance and chunks scoring below 0.3 are filtered out. On sparse or niche queries this may return no past context.
- Supercharge mode deduplicates by source URL but not by content - if the same content is served from multiple URLs it may be stored multiple times.
- The evaluation pipeline cannot delete `chroma_db` while the MCP vector store server is running on Windows due to file locking. Evaluation chunks are added to the live store and cleaned up by TTL eviction after 7 days. To reset immediately, stop the MCP vector store server and delete `chroma_db/` manually.
- arXiv search sorts by submission date so results are recent but may not always be the most relevant to the query. A relevance-sorted fallback could be added as a future improvement.
- arXiv paper summaries are truncated to 500 characters before chunking - full abstracts are longer and truncation may lose important context.
- arXiv API requires a descriptive User-Agent header and HTTPS - HTTP requests return a 301 redirect with empty body. The server uses `follow_redirects=True` and `https://` to handle this.
- arXiv API uses exact phrase matching with quoted terms (`ti:"topic" OR abs:"topic"`). Very short or generic topics may return fewer results than expected.
- The A2A pipeline calls agents via HTTP on port 8004 — each agent call adds network overhead compared to the direct sequential pipeline. For simple queries the fixed pipeline in `agent_pipeline.py` will be faster.
- The A2A critic re-write is triggered by keyword matching ("missing", "incorrect", "inaccurate", "unclear", "vague") in the feedback text. Substantive critic feedback that doesn't contain these words will not trigger a revision.
- The A2A analyst re-research is triggered by weak phrase detection ("no information", "insufficient" etc.) in the insights. If the analyst produces plausible-sounding but shallow insights without these phrases, re-research will not be triggered.
- The A2A pipeline requires all six terminals running (Ollama, three MCP servers, A2A agent server, and optionally the FastAPI main API) — more moving parts than the sequential pipeline which only needs Ollama and the three MCP servers.
- The reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) loads on first use and adds ~100-200ms per search on CPU. If unavailable, the system falls back to cosine similarity scoring automatically.
- The SSE stream runs each agent sequentially in a thread executor — if one agent is slow (e.g. writer with a large context), the stream will pause at that step with no intermediate updates until the agent completes.
- The Gemini free tier allows 20 requests per day per model on gemini-2.5-flash. With only the analyst, writer and critic calling Gemini (3 calls per pipeline run), this gives approximately 6 full pipeline runs per day before hitting the daily limit. Running `run_all.py` with Gemini consumes 7 calls in a single pass due to individual agent checks, so no more than 2-3 full verification runs per day are possible on the free tier. Enable billing on Google AI Studio for significantly higher limits.
- The subscription scheduler has no built-in recurring trigger — it must be invoked manually via `python scripts/run_subscriptions.py` or the `/subscriptions/run` API endpoint, or wired into an external scheduler (cron, Windows Task Scheduler, APScheduler). Running it more frequently than the subscription interval is safe since due-checking skips subscriptions that are not yet due.
- Subscription reports are delivered as plain text. Markdown formatting in the report is preserved in the payload but may not render in email clients unless the delivery helper is extended to send `text/html` with a markdown-to-HTML conversion step.
- Subscriptions inherit the global LLM_PROVIDER setting from .env — there is no per-subscription provider setting. The .env file is read live on each pipeline run, so switching providers via the Streamlit sidebar or editing .env directly takes effect on the next scheduled run without restarting the server.

---

## Possible next steps

### Quick Hits

- Make Gemini model names configurable via environment variables — replace the hardcoded gemini-2.5-flash in analyst.py, writer.py, and critic.py with os.environ.get("GEMINI_MODEL_QUALITY", "gemini-2.5-flash"), and the hardcoded gemini-2.0-flash-lite in planner.py, researcher.py, and graph_builder.py with os.environ.get("GEMINI_MODEL_LITE", "gemini-2.0-flash-lite"). Both variables are already documented in .env.example. Six one-line changes across six agent files.
- Fix start.py --stop to kill servers by port rather than by saved PID — use netstat on Windows and lsof on Mac/Linux to find the PID currently occupying each known port (8000, 8001, 8002, 8003, 8501, 8004) and kill it directly, so --stop reliably terminates all servers regardless of whether their PIDs were saved in .server_pids.json. Keep PID-based cleanup as a fallback for any ports not found via netstat.
- Fix silent exception swallowing in MCPClient.call_tool — add a print or logger call before returning the empty list so failures are distinguishable from genuine empty results. One line change in mcp_client.py.
- Fix the SERVER_PORTS fallback in MCPClient.call_tool — remove the default fallback to 8001 and raise a KeyError or log an explicit warning when an unrecognised server name is passed, so misconfigured calls fail loudly rather than silently routing to the wrong server. One line change in mcp_client.py.
- Fix the thread-safety bug in the vector store compare endpoint — replace the pattern of setting store.reranker = None directly on the live store object with a flag passed into the search method, so concurrent requests during a compare call do not silently lose the reranker. Change confined to vector_store_server.py and vector_store.py.
- Standardise chunking across all source servers — arXiv results are currently returned as single truncated documents while web search results are chunked at 200 characters with 50-character overlap inside the server, making retrieval inconsistent across sources. Move chunking out of web_search_server.py and into the researcher agent so all sources — web search, arXiv, and any future server — are chunked uniformly after merging. Changes touch researcher.py, web_search_server.py, and arxiv_server.py.
- Add a ping or health endpoint to each MCP server and a startup check in MCPClient — each server exposes a GET /health route returning 200, and the client checks reachability on initialisation and logs a warning per unavailable server rather than failing silently at query time. Changes touch mcp_client.py and each of the three server files.
- Pass planner sub-tasks into researcher.py as additional retrieval queries alongside the original question; run ChromaDB retrieval for each, union the candidate chunks, and pass the full set to the reranker. Requires no schema changes or new servers. Do this alongside the asyncio parallelism change since both touch the same researcher.py call flow.
- Extend the multi-query approach to DuckDuckGo and arXiv MCP calls; run one search per planner sub-task and merge results before chunking into ChromaDB. Deduplication by source URL already exists in the web search server so merging is straightforward.
- Make the relevance score threshold configurable via environment variable.
- Add an optional low-temperature Ollama call in researcher.py before retrieval that rephrases the research question into three to five alternative queries; union these with the planner sub-tasks and use the combined set for both ChromaDB and web search retrieval. Gate behind an environment variable so it can be toggled off. Add after multi-query retrieval is confirmed working since it builds on the same multi-query path.
- Add a built-in recurring trigger to the subscription scheduler — replace the manual `python scripts/run_subscriptions.py` invocation with an APScheduler `BackgroundScheduler` that starts automatically when the FastAPI app launches, running `run_due_subscriptions()` on a configurable interval (default every 6 hours, set via `SCHEDULER_INTERVAL_HOURS` in `.env`). Add the scheduler startup to the FastAPI `lifespan` context manager in `api/main.py` so it starts and stops cleanly with the server. Add `SCHEDULER_INTERVAL_HOURS=6` to `.env.example`. Changes confined to `api/main.py` and `src/scheduler/scheduler.py` with no changes to the subscription store or delivery logic.
- Add a per-subscription LLM provider field to the subscription store — extend the subscription schema with an optional `llm_provider` field (`"ollama"` or `"gemini"`) stored in `subscriptions.json`. Before running the pipeline for a subscription, temporarily set `os.environ["LLM_PROVIDER"]` to the subscription's provider value and restore it after the run, so each subscription can use a different provider independently of the global `.env` setting. Changes confined to `subscription_store.py`, `scheduler.py`, and the `/subscriptions` POST endpoint schema in `api/main.py`.
- Expose the per-subscription LLM provider field in the Streamlit subscriptions UI — once the `llm_provider` field is added to the subscription schema, add a provider selector (Ollama/Gemini) to the subscription creation form in the sidebar so users can choose per-subscription which provider generates the report. Display the stored provider alongside the other subscription details in the subscription card view. One additional field in the form and one additional column in the card display, both reading from and writing to the existing `/subscriptions` POST endpoint. Depends on the per-subscription LLM provider backend change being implemented first.
- Reframe the LoRA fine-tuning section as a standalone experimental track rather than a pipeline component — move it to a separate FINETUNING.md, add a note at the top clarifying it demonstrates the fine-tuning workflow rather than providing a production-quality writer, remove the LoRA writer from the stack table to avoid implying parity with other components, and retain the evaluation table since it honestly illustrates the impact of model scale and training data size. No code changes required.

### Bigger Changes


- Subscription management in the Streamlit UI — add a subscriptions panel to the sidebar in `streamlit_app.py` below the provider selector. On load, call `GET /subscriptions` and display each subscription as a card showing topic, frequency, delivery method, last run timestamp and status (active/paused). Add a form to create a new subscription with fields for topic, frequency (daily/weekly/monthly), delivery method (log/email/slack/discord) and delivery target. Add pause, resume and delete buttons per subscription that call the corresponding API endpoints. Add a Run Now button that calls `POST /subscriptions/run` to trigger the scheduler manually and shows a spinner while it runs. Keep all subscription API calls going through the existing FastAPI layer rather than calling `subscription_store.py` directly from the UI, so the UI remains decoupled from the storage layer. Changes confined to `ui/streamlit_app.py` with no backend changes required since all endpoints already exist.
- Source citations in report — thread source URLs from retrieved chunks through the pipeline to the writer, adding a References section to the report so every claim is traceable to a specific web page or arXiv paper.
- Supercharge mode in Streamlit UI — add a sidebar panel showing the current state of the vector database (total chunks stored, topics covered, latest ingestion timestamp) with a button to run supercharge on any topic directly from the UI before triggering the main pipeline, so users can pre-populate memory without leaving the interface.
- Latency optimisation — run web search and arXiv calls in parallel using asyncio rather than sequentially, and cache frequent ChromaDB queries with a short TTL.
- Supercharge-aware researcher — before running web search and arXiv, check ChromaDB for existing chunks on the topic using a coverage threshold; if sufficient pre-populated content exists from a supercharge run, skip the researcher entirely and pass cached chunks directly to the analyst, saving web search calls and pipeline time.
- Advanced chunking strategies — replace fixed-size character chunking with semantic chunking on sentence or paragraph boundaries, with larger chunk sizes for arXiv abstracts which are denser than web snippets.
- Report grounding improvement — add a post-writer step that cross-references claims in the report against the retrieved chunks and flags or removes any statement not supported by the source documents, reducing hallucination in the final output.
- Quantitative critic scoring for A2A re-write decisions — replace the keyword-matching trigger in the A2A critic with a structured scoring step. After the critic produces its qualitative feedback, run a second low-temperature LLM call that asks the model to score the report on a 1-10 scale across four dimensions (relevance, completeness, clarity, accuracy) and return JSON. Use the average score as the re-write threshold — if it falls below a configurable value (default 7, set via `CRITIC_REWRITE_THRESHOLD` in `.env`) the A2A coordinator requests a revision, passing both the qualitative feedback and the per-dimension scores to the writer so it knows specifically what to improve. Store the scores in the pipeline result dict alongside `critic_feedback` so they are visible in the Streamlit UI and API response. This reuses the same LLM-as-judge scoring logic already built in `src/evaluation/evaluator.py` — the evaluator can be called directly rather than duplicating the prompt, making this a relatively contained change across `a2a_pipeline.py`, `agent_server.py`, and the Streamlit result rendering in `streamlit_app.py`.
- LangGraph pipeline — replace the sequential for-loop in `agent_pipeline.py` with a LangGraph stateful graph where each agent is a node and edges represent the flow between them. Define a shared state schema (question, tasks, documents, insights, entities, report, critic_feedback) passed between nodes rather than returned as a dict at the end. This makes the pipeline easier to visualise, debug, and extend — adding a new agent is adding a node and two edges rather than editing the run method. The A2A re-invocation logic (re-research if insights weak, rewrite if critic flags issues) becomes explicit conditional edges with clear branching conditions rather than keyword matching on strings, making the dynamic behaviour inspectable and testable. Implement in a new `src/workflow/langgraph_pipeline.py` alongside the existing `agent_pipeline.py` so both can run independently — the existing pipeline remains the default and LangGraph becomes an opt-in via an environment variable or CLI flag. Changes confined to `src/workflow/` with no agent, MCP, or API changes required.
- LangGraph A2A replacement — once the LangGraph pipeline is stable, replace the A2A HTTP endpoint layer in `src/a2a/` with LangGraph's native conditional edge branching. The A2A coordinator currently calls agents via HTTP on port 8004 adding network overhead on every agent invocation — LangGraph runs everything in-process with state passed directly between nodes, eliminating the HTTP round-trips. The critic re-write trigger and analyst re-research trigger become graph conditions checked at edge evaluation time with configurable thresholds (e.g. the quantitative critic score from `CRITIC_REWRITE_THRESHOLD`) rather than string matching. Implement as `src/workflow/langgraph_a2a_pipeline.py`. Depends on the basic LangGraph pipeline above being confirmed working first.
- Graph-informed RAG retrieval — query Neo4j for entities related to the research topic before ChromaDB retrieval, using known entity relationships to expand the search context and surface more relevant chunks.
- Add an MCP server for Neo4j to fully decouple the knowledge graph from agents, consistent with the rest of the MCP architecture.
- Longer context window — switch to a model with a larger context window (e.g. llama3.1:8b or Gemini) so the writer receives the full analyst insights and all retrieved chunks rather than a truncated version, producing more complete and detailed reports.
- Retrieval-augmented prompt strategy — the analyst currently uses naive stuffing, injecting all retrieved chunks directly into a single prompt. As vector memory grows this will silently overflow the context window. Add a map-reduce fallback: if the total retrieved content exceeds a token threshold, split chunks into batches, run the analyst prompt independently on each batch to extract partial insights, then run a final synthesis prompt over the partial insights. This avoids truncation without requiring a larger context window and fits naturally into analyst.py with no pipeline changes. A refine pattern (iteratively updating a running summary with each new chunk) is an alternative worth considering if partial insight quality is poor, but map-reduce is simpler to implement and debug first.
- Multi-run synthesis — run the pipeline on the same topic multiple times with different search queries and synthesise the results into a single consolidated report, reducing the impact of any single poor web search result.
- Fine-tune a larger model (e.g. llama3.2) with more training data for meaningful quality improvement, making the fine-tuning track a genuine alternative to Ollama rather than a workflow demonstration.
- Visualise the knowledge graph in the Streamlit UI.
- Entity-grounded chunking — tag ChromaDB chunks with entities extracted by the graph builder at storage time, enabling hybrid retrieval by entity and semantic similarity.
- Report personalisation — allow the user to specify report length, audience (technical vs non-technical) and focus areas via the Streamlit UI before running the pipeline, passing these as constraints to the writer prompt.
- Expanded query caching — cache LLM-generated query expansions in a lightweight key-value store keyed by a hash of the original question; repeated queries on similar topics skip the expansion LLM call and reuse prior variants. Only worth adding once query expansion is confirmed to improve retrieval scores.
- Hierarchical retrieval — when arXiv or supercharge ingestion is extended to full-length documents rather than truncated snippets, store a summary embedding per document alongside fine-grained chunk embeddings; use summary embeddings for first-pass recall and chunk embeddings for precision before passing candidates to the existing reranker. Not worth adding until document length justifies the added layer — current 200-character chunks from short snippets do not.
- Add DPO (Direct Preference Optimisation) training using critic feedback as preferred/rejected pairs — fits naturally into the existing LoRA training pipeline. Depends on fine-tuning a larger model with sufficient training data first.
- Query expansion evaluation — once multi-query retrieval is in place, extend the retrieval quality metric to benchmark query expansion on versus off, measuring whether expanded query variants surface chunks that the original single query misses.
- Evaluation framework expansion — extend the LLM-as-judge evaluator to score RAG retrieval quality separately from report quality, and add reproducibility metrics by measuring score variance across repeated runs.
- Hallucination reduction — add a fact-checking step between the analyst and writer that cross-references claims against retrieved source documents before the report is written. Overlaps significantly with the report grounding post-writer step; implement one before considering the other.
- Monitoring and observability — add structured logging per agent with timestamps, token counts and latency metrics, exportable to a dashboard.
- API authentication for deployment.
- Secrets manager integration — replace `GEMINI_API_KEY` in `.env` with a runtime fetch from a secrets manager (AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault) for production deployments. Add a `src/config/secrets.py` helper that calls the relevant SDK on startup and falls back to `os.environ` for local development, so `.env` continues to work unchanged for contributors running locally. The rest of the codebase reads the key through `secrets.py` rather than `os.environ.get` directly, meaning the switch from local to production requires no code changes beyond setting the secrets manager backend via an environment variable. Vault or AWS Secrets Manager would be the most common choices for a self-hosted or AWS-deployed version of this stack respectively.
- Gemini quota tracker — build a lightweight daily usage tracker in `src/config/gemini_quota.py` that records request counts per model in a local JSON file (e.g. `gemini_quota.json`), resets automatically when the date changes, and exposes a `remaining(model)` function returning how many requests are left against the free tier limit for that model. Surface the per-model counts in the Streamlit sidebar so the user can see remaining capacity before running the pipeline. No external dependencies required beyond the standard library — the tracker reads and writes a single JSON file on each API call via a thin wrapper around `_call_gemini` in `base_agent.py`.
- Gemini model fallback orchestration — before each Gemini agent call, check `gemini_quota.remaining(model)` and if the remaining count would not cover the full pipeline run, automatically promote to the next available model (e.g. gemini-2.5-flash → gemini-2.0-flash-lite → gemini-2.0-flash) by updating `self.gemini_model` at call time rather than at init. If all tracked models are exhausted for the day, fall back to Ollama automatically and print a clear message: "All Gemini model daily limits reached — falling back to Ollama". Implement in `base_agent._call_gemini` with a priority list of fallback models defined in `.env.example` as `GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.0-flash-lite,gemini-2.0-flash`. Depends on the quota tracker above being implemented first.
- Prompt injection detection — add input sanitisation in the researcher and analyst that detects and rejects retrieved content containing instruction-like patterns before prompt injection.
- PII detection and filtering — add a pre-storage filter in the MCP vector store server that detects and redacts personally identifiable information before chunks are written to ChromaDB.
- Multilingual support — detect document language before chunking, translate non-English content before embedding to improve retrieval quality across languages.
- Traceability and compliance — attach source URLs and retrieval timestamps to every claim in the final report, and maintain an audit log of which chunks were retrieved and injected into each agent prompt per run.

---


## License

MIT
