# Sydney Water information assistant

Deployment instructions: [DEPLOYMENT.md](DEPLOYMENT.md). Model and credentials come exclusively from `.env`; never commit that file.

**Source update (21 October 2026):** 42 webpages / 342 sections are now indexed. See the [source import report](data/SOURCE_IMPORT_REPORT.md) for excluded URLs and validation. Expanded-corpus retrieval scored 20/27; four offline tests currently fail. Earlier evaluation figures below describe the smaller corpus.

Independent student proof of concept, **not an official Sydney Water service**. The local Streamlit demo answers public residential billing, payment and assistance questions using one LangChain agent and two read-only MCP tools.

**Acceptance is not yet complete.** The application runs, but live-model reliability failures and human semantic review remain outstanding. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for actual results; a valid citation reference does not prove that every claim follows from its source. Delivery checks: 51 local tests passed; both 36-scenario agent/baseline runs completed. Frozen retrieval scored 88.9%, below the 90% target, and source review found incomplete answers.

## Run the installed project

From PowerShell in this directory:

```powershell
.\run.ps1 python knowledge.py status
.\run.ps1 python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Try “What does the usage charge mean?”, “What ways can I pay my bill?”, or “My bill went up and I am struggling to pay it.” Ask a follow-up, expand **Sources used**, and use **New conversation** to reset.

Do not submit real personal, account or payment information. Input checks catch common patterns, not every possible form of personal information. Only approved, validated source links are clickable; model text is displayed as plain text after validation.

## Install on Windows

Python 3.12 is required by the locked application. The bootstrap below can use an existing Python installation to install uv, which downloads a project-local Python 3.12 runtime:

```powershell
python -m pip install --target .bootstrap uv
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.runtime'
$env:UV_SYSTEM_CERTS = 'true'
.\.bootstrap\bin\uv.exe sync --python 3.12 --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The current workspace is already installed. **Do not replace its existing `.env`.** Keys are read from the environment and the ignored `.env`; they are not committed in application code. Model credentials and selection are read only from the project `.env`; Windows User/Machine keys are not used. The configured Groq model is supported directly. The script preserves TLS verification using Windows trusted certificates.

Configuration names:

```dotenv
LLM_PROVIDER=groq
GROQ_MODEL=YOUR_CONFIGURED_MODEL
GROQ_API_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
TRACE_CONTENT=false
LLM_THINKING_LEVEL=low
```

Use a free-tier project with paid billing disabled. The project owner confirmed that setting for the supplied Gemini key. The originally planned Gemini 3.8 Flash returned 503/timeouts during integration; the explicitly configured 3.5 Flash-Lite passed the real integration tests. There is no automatic provider switch or retry. The current configuration uses the Groq model specified in `.env`; Gemini results below are historical. There is no automatic fallback.

## Rebuild and review sources

Stop Streamlit before replacing its corpus:

```powershell
.\run.ps1 python knowledge.py ingest
```

Inspect the eight HTML snapshots in `data/snapshots/` and extracted sections in `data/review/`. Compare conditions, tables, dates and contact/navigation text with the official pages. Only after that review, record the actual reviewer:

```powershell
.\run.ps1 python knowledge.py review --ids read_bill payment_options payment_help bill_payment meter_readings residential_prices my_account contact --reviewer YOUR_REVIEWER_NAME
.\run.ps1 python knowledge.py status
```

The supplied corpus records **agent-assisted-source-comparison**, not human approval. Runtime readiness rejects missing reviews or reviews older than 30 days. Rebuilding changes the corpus version and invalidates previous conversational evidence. A failed rebuild preserves the existing database; generated review files from a partial rebuild must be inspected again before recording review.

The eight exact official URLs are in `data/sources.json`. Ingestion checks redirect destinations and canonical hosts, removes executable/advertising material, preserves whole paragraphs and tables, and indexes English text with SQLite FTS5/Porter stemming. Customer conversations never fetch the web.

## Test and evaluate

Default tests make **no model API calls**. They exercise real local stdio MCP subprocesses with scripted model responses, deterministic boundaries and Streamlit AppTest:

```powershell
.\run.ps1 python -m pytest -q
```

The explicit live integration tests consume free provider quota:

```powershell
.\run.ps1 python -m pytest -q -m live
.\run.ps1 python evaluate.py --suite smoke --output data/my_smoke_results.jsonl
.\run.ps1 python evaluate.py --suite retrieval --split development --output data/my_retrieval_results.json
```

The fixed final dataset has 36 scenarios, 24 development and 12 holdout. The roadmap also requires M03 during the earlier memory stage, so that case is not an unseen holdout. The final comparison commands are:

```powershell
.\run.ps1 python evaluate.py --suite final --mode agent --trace-content --output data/my_agent_results.jsonl
.\run.ps1 python evaluate.py --suite final --mode baseline --output data/my_baseline_results.jsonl
.\run.ps1 python evaluate.py --suite report --agent-results data/my_agent_results.jsonl --baseline-results data/my_baseline_results.jsonl --output EVALUATION_REPORT.md
```

Use `--ids M01 M02 M03 M04` for a focused run. Runs wait 15 seconds between turns by default (`--delay` adjusts this). Results append after each completed scenario; rerunning the same command skips completed IDs. A quota response stops the run and records the incomplete attempt separately. A different model, prompt, corpus or dataset requires a new output filename. An interrupted incomplete scenario may consume quota again on resume. The baseline performs exactly one MCP search and one structured model call per accepted turn, using the same corpus, provider, output/citation gate and history limits; it has no agent loop.

Response-kind and source-coverage checks are **not accuracy scores**. Review each answer against its cited passages and the dataset's required facts/prohibited claims. Record missing conditions, unsupported assertions and safe fallbacks separately. Human claim-support and correctness review is required for final acceptance. Do not count every safe fallback as a correct answer.

## Architecture and limits

`Streamlit → create_agent / LangGraph → MCP search_docs or read_section → validated structured answer → trusted citations`

- One `create_agent`, `ToolStrategy(Answer, handle_errors=False)`, and a session-local `InMemorySaver` with random thread ID. Only the new user message is submitted; LangGraph manages the successful history.
- One local stdio MCP server. `search_docs` uses literal, bounded FTS queries; `read_section` accepts only a corpus section ID. Tools cannot fetch URLs, write files, execute commands or access accounts.
- Citations resolve only against validated tool artifacts delivered in the current conversation and exact allowlisted URLs. No LLM verifier, planner, repair call, vector database or provenance ledger.
- Four model requests and six tool calls per turn; 25-second model, five-second tool and 100-second overall deadlines. Approximately 6,000 input tokens, 2,048 output tokens, 2,000 input characters, 20 turns and 30-minute idle session limits.
- Invalid candidates, interrupted runs and exhausted budgets reset the affected saver/thread. No checkpoint rollback or message reconstruction. When all searches returned no results, exhausting the model budget produces a fixed insufficient-evidence message and reset.
- No long-term memory. A new conversation clears local session state, not previously exported cloud traces. Process restart discards all in-memory conversations.

The model occasionally repeats searches, produces rejected citations or fails to finish within the budget. These failures are surfaced and retained in evaluation; the limits are not increased to hide them. Exact-source validation cannot automatically detect unsupported paraphrases or omitted conditions.

The final allowed model call exposes only the structured Answer tool, reserving space to finish or report insufficient evidence. Repeated identical passages within a turn are represented by their already-delivered evidence IDs; full artifacts and the earlier source text remain in LangGraph state. This reduces repeated input without summarisation calls or discarding source conditions. `LLM_THINKING_LEVEL` is fixed for each evaluation and participates in session reset/configuration checks; the selected setting remains `low`.

## Langfuse

The supported `langfuse.langchain.CallbackHandler` captures model/tool spans inside one outer request observation. Default metadata-only masking excludes input/output, raw errors and unknown metadata. It retains controlled outcome/source IDs, call counts, latency, available usage and random session grouping. The SDK's own project/scope metadata is separate from conversation content.

`--trace-content` enables redacted content **only for synthetic evaluation**. Known keys, common sensitive patterns and reasoning/thinking fields are removed by the supported export hook. It does not modify live model protocol messages. The UI reports configuration status; export is asynchronous, so configuration alone is not proof of delivery. Tracing setup failures do not rerun or block the agent. Inspect the recorded cloud trace to verify delivery. Langfuse list-price estimates are not evidence of actual billing.

## Repository

| File | Responsibility |
|---|---|
| `app.py` | Streamlit chat, validated rendering and per-session UI state |
| `agent.py` | Single agent, checkpoint lifecycle, boundaries, citations and tracing |
| `mcp_server.py` | Two read-only MCP tool definitions |
| `knowledge.py` | Approved-source ingestion, review metadata and SQLite retrieval |
| `schemas.py` | Strict tool evidence and final-answer contracts |
| `evaluate.py` | Smoke/retrieval/final runners and one-shot baseline |
| `test_system.py` | Offline tests and explicitly marked live integration tests |
| `data/sources.json`, `data/eval_cases.jsonl` | Source allowlist and fixed labelled scenarios |
| `pyproject.toml`, `uv.lock`, `run.ps1` | Dependencies, reproducible runtime and environment loading |

No API backend, Docker, cloud database or additional agent is used. The authoritative design remains `IMPLEMENTATION_PLAN.md`; deviations, test evidence and incomplete gates are tracked in `IMPLEMENTATION_STATUS.md`.
