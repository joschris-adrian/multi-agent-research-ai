# Multi-Agent Research Assistant

I built this to explore how multiple LLM agents can collaborate on a research task - each one handling a specific job rather than dumping everything into a single prompt.

The system runs fully locally using Ollama, so no API keys or costs involved.

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
| LLM             | Ollama (llama3.2)               |
| Agent pipeline  | Custom multi-agent architecture |
| Web search      | DuckDuckGo via `ddgs` + MCP     |
| Vector memory   | ChromaDB via MCP                |
| MCP servers     | FastAPI (vector store: 8001, web search: 8002, arXiv: 8003) |
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
│   └── supercharge.py      # Bulk ingest documents into ChromaDB without running pipeline
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
│   │   │   └── mcp_client.py          # HTTP client used by all agents
│   │   └── servers/
│   │       ├── vector_store_server.py # ChromaDB exposed as MCP server
│   │       ├── web_search_server.py   # DuckDuckGo exposed as MCP server
│   │       └── arxiv_server.py        # arXiv API exposed as MCP server
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

**Option 2a - Single command startup:**
```bash
# start all servers (Ollama, MCP servers, FastAPI, Streamlit)
python start.py

# start including A2A pipeline server
python start.py --a2a

# stop all servers started by start.py
python start.py --stop
```
Servers are started as background processes with health checks. PIDs are tracked in `.server_pids.json` and cleaned up on stop.


**Option 2b - API + UI (five terminals):**
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
- MCP servers must be running separately for agents to access web search and vector memory. Vector store runs on port 8001, web search on port 8002. The analyst degrades gracefully if unavailable, but the researcher will return empty results.
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

---

## Possible next steps

### Quick Hits

- Fix silent exception swallowing in MCPClient.call_tool — add a print or logger call before returning the empty list so failures are distinguishable from genuine empty results. One line change in mcp_client.py.
- Fix the SERVER_PORTS fallback in MCPClient.call_tool — remove the default fallback to 8001 and raise a KeyError or log an explicit warning when an unrecognised server name is passed, so misconfigured calls fail loudly rather than silently routing to the wrong server. One line change in mcp_client.py.
- Fix the thread-safety bug in the vector store compare endpoint — replace the pattern of setting store.reranker = None directly on the live store object with a flag passed into the search method, so concurrent requests during a compare call do not silently lose the reranker. Change confined to vector_store_server.py and vector_store.py.
- Standardise chunking across all source servers — arXiv results are currently returned as single truncated documents while web search results are chunked at 200 characters with 50-character overlap inside the server, making retrieval inconsistent across sources. Move chunking out of web_search_server.py and into the researcher agent so all sources — web search, arXiv, and any future server — are chunked uniformly after merging. Changes touch researcher.py, web_search_server.py, and arxiv_server.py.
- Add a ping or health endpoint to each MCP server and a startup check in MCPClient — each server exposes a GET /health route returning 200, and the client checks reachability on initialisation and logs a warning per unavailable server rather than failing silently at query time. Changes touch mcp_client.py and each of the three server files.
- Pass planner sub-tasks into researcher.py as additional retrieval queries alongside the original question; run ChromaDB retrieval for each, union the candidate chunks, and pass the full set to the reranker. Requires no schema changes or new servers. Do this alongside the asyncio parallelism change since both touch the same researcher.py call flow.
- Extend the multi-query approach to DuckDuckGo and arXiv MCP calls; run one search per planner sub-task and merge results before chunking into ChromaDB. Deduplication by source URL already exists in the web search server so merging is straightforward.
- Make the relevance score threshold configurable via environment variable.
- Add an optional low-temperature Ollama call in researcher.py before retrieval that rephrases the research question into three to five alternative queries; union these with the planner sub-tasks and use the combined set for both ChromaDB and web search retrieval. Gate behind an environment variable so it can be toggled off. Add after multi-query retrieval is confirmed working since it builds on the same multi-query path.
- Add a download button in the Streamlit UI to save the final report as a markdown file.
- Reframe the LoRA fine-tuning section as a standalone experimental track rather than a pipeline component — move it to a separate FINETUNING.md, add a note at the top clarifying it demonstrates the fine-tuning workflow rather than providing a production-quality writer, remove the LoRA writer from the stack table to avoid implying parity with other components, and retain the evaluation table since it honestly illustrates the impact of model scale and training data size. No code changes required.

### Bigger Changes

- Make the vector store server protocol-compliant — implement a tools/list handler returning tool definitions with name, description, and input schema, and a tools/call handler accepting the standardised MCP invocation envelope (tool name, arguments object, invocation ID) and returning typed content blocks with isError flags and error metadata. Use SSE transport via the official Python MCP SDK so the FastAPI server stays as an HTTP server with the MCP protocol layered on top. Core ChromaDB logic stays unchanged.
- Make the web search server protocol-compliant — apply the same tools/list and tools/call pattern as the vector store server. The web search server is the simplest of the three so use it to validate the pattern established in the previous step before moving to arXiv.
- Make the arXiv server protocol-compliant — apply the same tools/list and tools/call pattern. Confirm that the SSE transport and response envelope work consistently across all three servers before proceeding to client replacement.
Replace the bespoke MCPClient with a standard MCP-compatible client — once all three servers are compliant, drop mcp_client.py and replace it with a standard client using the official Python MCP SDK. All agents that currently call MCPClient.call_tool will route through the standard client instead.
- Google Gemini integration — add configurable LLM provider support in base_agent.py via LLM_PROVIDER environment variable, switching between Ollama (local, free) and Gemini API (cloud, free tier via Google AI Studio). Assign gemini-2.0-flash to analyst, writer and critic where output quality matters, and gemini-1.5-flash-8b to planner, researcher and graph builder where the task is simple enough that a smaller model suffices. Add a 4-second inter-agent delay when using Gemini to stay within the free tier 15 RPM limit. No agent, pipeline or MCP code changes required beyond base_agent.py and per-agent model defaults.
- Source citations in report — thread source URLs from retrieved chunks through the pipeline to the writer, adding a References section to the report so every claim is traceable to a specific web page or arXiv paper.
- Supercharge mode in Streamlit UI — add a sidebar panel showing the current state of the vector database (total chunks stored, topics covered, latest ingestion timestamp) with a button to run supercharge on any topic directly from the UI before triggering the main pipeline, so users can pre-populate memory without leaving the interface.
- Error recovery — implement checkpoint saving so a failed pipeline run can resume from the last successful agent rather than restarting from the planner.
- Latency optimisation — run web search and arXiv calls in parallel using asyncio rather than sequentially, and cache frequent ChromaDB queries with a short TTL.
- Supercharge-aware researcher — before running web search and arXiv, check ChromaDB for existing chunks on the topic using a coverage threshold; if sufficient pre-populated content exists from a supercharge run, skip the researcher entirely and pass cached chunks directly to the analyst, saving web search calls and pipeline time.
- Advanced chunking strategies — replace fixed-size character chunking with semantic chunking on sentence or paragraph boundaries, with larger chunk sizes for arXiv abstracts which are denser than web snippets.
- Report grounding improvement — add a post-writer step that cross-references claims in the report against the retrieved chunks and flags or removes any statement not supported by the source documents, reducing hallucination in the final output.
- Graph-informed RAG retrieval — query Neo4j for entities related to the research topic before ChromaDB retrieval, using known entity relationships to expand the search context and surface more relevant chunks.
- Add an MCP server for Neo4j to fully decouple the knowledge graph from agents, consistent with the rest of the MCP architecture.
- Longer context window — switch to a model with a larger context window (e.g. llama3.1:8b or Gemini) so the writer receives the full analyst insights and all retrieved chunks rather than a truncated version, producing more complete and detailed reports.
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
- Prompt injection detection — add input sanitisation in the researcher and analyst that detects and rejects retrieved content containing instruction-like patterns before prompt injection.
- PII detection and filtering — add a pre-storage filter in the MCP vector store server that detects and redacts personally identifiable information before chunks are written to ChromaDB.
- Multilingual support — detect document language before chunking, translate non-English content before embedding to improve retrieval quality across languages.
- Traceability and compliance — attach source URLs and retrieval timestamps to every claim in the final report, and maintain an audit log of which chunks were retrieved and injected into each agent prompt per run.

---


## License

MIT
