# VectorRAG — 10-Q Filing Q&A (OpenAI + Chroma)

A production-minded **Retrieval-Augmented Generation** service that ingests a
company's **10-Q quarterly filings** and answers natural-language questions about
them using **OpenAI** models, grounded in the filing text with **citations**.

> Ask *"What was revenue last quarter and how did it change year over year?"* and
> get an answer backed by the exact passages from the filing — or an honest
> *"I don't have enough information"* when the filings don't cover it.

---

## Why this design

| Concern | How it's addressed |
|---|---|
| **Hallucinations** | Strict grounded system prompt ("answer ONLY from context"), `temperature=0`, mandatory `[S#]` citations, and **abstention** when no relevant chunk is retrieved (`min_relevance_score`). |
| **Token / cost control** | `tiktoken`-based token counting everywhere: token-bounded chunking, a hard **context budget** (`max_context_tokens`), capped output (`max_output_tokens`), and batched embeddings. |
| **Conversation memory** | Rolling, **token-bounded** history (`max_history_tokens`) so multi-turn chat stays in-budget. |
| **Security** | Input sanitization + length limits, **prompt-injection** heuristics on both user input and retrieved text, retrieved content **neutralized** and treated as untrusted data, **PII/secret redaction** in logs, API key from **AWS Secrets Manager** (never on disk in prod). |
| **Reliability** | Retries with exponential backoff (`tenacity`) on all OpenAI calls; idempotent re-ingest (`upsert` by chunk id). |
| **Observability** | Structured JSON logs (CloudWatch-friendly) with token counts, latency, and source ids — no secrets/prompts logged. |

---

## Architecture

```
                 ingest (offline)                          query (online)
  ┌────────────┐   ┌─────────┐   ┌──────────┐      ┌──────────────┐
  │ 10-Q files │──▶│ loader  │──▶│ chunker  │      │  user query  │
  │ pdf/html/  │   │ (pages) │   │ (tokens) │      └──────┬───────┘
  │ txt        │   └─────────┘   └────┬─────┘             │ sanitize + injection check
  └────────────┘                      │ embed             ▼
                                      ▼            ┌──────────────┐
                              ┌───────────────┐    │  Retriever   │ embed query
                              │ OpenAI embed  │    │  top-k +     │──────────────┐
                              └───────┬───────┘    │  budget      │              ▼
                                      ▼            └──────┬───────┘      ┌────────────────┐
                              ┌───────────────┐          │              │ OpenAI embed   │
                              │  Chroma (DB)  │◀─────────┴──────────────│  + Chroma search│
                              └───────────────┘   retrieved chunks      └────────────────┘
                                                          │
                                                          ▼  grounded prompt + citations
                                                  ┌────────────────┐
                                                  │ OpenAI chat    │──▶ answer + [S#] sources
                                                  └────────────────┘
```

Code map (under `src/vectorrag/`):

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings (env / `.env` / Secrets Manager). |
| `ingest/loader.py` | Parse PDF / SEC EDGAR HTML / text into pages. |
| `ingest/chunker.py` | Token-aware overlapping chunking. |
| `ingest/pipeline.py` | files → chunks → embeddings → Chroma. |
| `embeddings.py` | Batched OpenAI embeddings + retry. |
| `vectorstore.py` | Chroma wrapper (cosine; precomputed vectors). |
| `retriever.py` | Query embed, search, relevance filter, context budget. |
| `llm.py` | OpenAI chat completion + retry. |
| `memory.py` | Token-bounded conversation history. |
| `security.py` | Sanitization, injection detection, PII redaction. |
| `prompts.py` | Grounded, citation-enforcing prompt templates. |
| `rag.py` | Orchestration / public `RAGEngine`. |
| `cli.py` | `vectorrag` command-line tool. |
| `api/app.py` | FastAPI service; `api/lambda_handler.py` Lambda adapter. |

---

## Prerequisites

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency and
environment management. uv also provisions the Python interpreter itself, so you
do **not** need a system Python.

- **uv** ≥ 0.5 — install:
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
- An **OpenAI API key**
- (Deploy only) Docker, AWS account, Terraform ≥ 1.6

---

## Setup

```bash
# 1. Create the locked environment (uv downloads Python 3.12 automatically,
#    builds .venv, and installs all deps from uv.lock).
uv sync

# 2. Configure
cp .env.example .env      # then edit .env and set OPENAI_API_KEY
```

That's it — no manual venv activation needed. Prefix commands with `uv run`
(e.g. `uv run vectorrag ...`), or activate `.venv` if you prefer.

> Dependencies live in `pyproject.toml`; exact versions are pinned in `uv.lock`
> (committed). After changing a dependency, run `uv lock` then `uv sync`.

---

## Usage

All commands run through `uv run` (or activate `.venv` and drop the prefix).

### 1. Ingest filings

Put 10-Q files (`.pdf`, `.htm/.html`, `.txt`) under `data/raw/` (a sample is
included), then:

```bash
uv run vectorrag ingest data/raw
# or a single file:
uv run vectorrag ingest data/raw/SAMPLE_acme_10q.txt
```

Need a real filing? Grab one from SEC EDGAR:

```bash
uv run python scripts/fetch_10q.py --url <edgar-document-url> --out data/raw/company_10q.htm
```

### 2. Ask a question

```bash
uv run vectorrag ask "What was total revenue for the quarter and how did it change year over year?"
```

```
Total revenue for the three months ended June 30, 2025 was $1,245.3 million, a 12%
increase from $1,112.0 million in the prior-year period [S1].

Sources:
  [S1] SAMPLE_acme_10q.txt, p.1 (score=0.71)
```

### 3. Interactive chat (with memory)

```bash
uv run vectorrag chat
```

### 4. Inspect the store

```bash
uv run vectorrag info
```

### 5. Run the HTTP API locally

```bash
uv run uvicorn api.app:app --reload --port 8000

curl -s localhost:8000/ask -H "content-type: application/json" \
  -d '{"question":"What was net income per diluted share?"}' | jq
```

`GET /health` → `{"status":"ok"}`.

---

## Configuration reference

All settings are env vars (prefix `VECTORRAG_`) — see `.env.example`. Highlights:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (local dev). |
| `VECTORRAG_CHAT_MODEL` | `gpt-4o-mini` | Chat model. |
| `VECTORRAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `VECTORRAG_TEMPERATURE` | `0.0` | Low = less hallucination. |
| `VECTORRAG_TOP_K` | `5` | Chunks retrieved per query. |
| `VECTORRAG_MIN_RELEVANCE_SCORE` | `0.20` | Below this → abstain. |
| `VECTORRAG_CHUNK_SIZE_TOKENS` | `500` | Chunk size. |
| `VECTORRAG_MAX_CONTEXT_TOKENS` | `6000` | Hard context budget. |
| `VECTORRAG_MAX_OUTPUT_TOKENS` | `800` | Answer length cap. |
| `VECTORRAG_MAX_HISTORY_TOKENS` | `2000` | Chat memory budget. |
| `VECTORRAG_CHROMA_DIR` | `./.chroma` | Vector store path. |
| `VECTORRAG_OPENAI_SECRET_NAME` | — | Secrets Manager secret (prod). |

---

## Testing & quality

The `dev` dependency group (pytest, ruff, mypy) is installed by `uv sync` already.

```bash
uv run pytest               # unit tests (no network — OpenAI/Chroma are mocked)
uv run ruff check src api tests
uv run mypy src
# or simply: make test / make lint
```

The test suite covers chunking, token math, security utilities, memory trimming,
and the RAG orchestration (including the **abstain-when-no-context** guardrail).

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full AWS Lambda guide: building the
container image, pushing to ECR, the **Terraform** stack (`infra/terraform/`), and
the CI/CD **pipeline** (GitHub Actions in `.github/workflows/` or AWS CodeBuild in
`infra/pipeline/buildspec.yml`).

---

## Limitations & notes

- This is **not investment advice**; it summarizes filing text and can still err.
  Always verify figures against the source passages it cites.
- Chroma persists to local disk; for Lambda the store is hydrated from S3 into
  `/tmp` at cold start (see DEPLOYMENT.md). For large corpora consider EFS or a
  managed vector DB.
- Ingestion is intended to run offline/batch; the Lambda function serves queries.
