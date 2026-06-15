# Architecture & Interview Prep

A low-level architectural walkthrough of **VectorRAG** — a vector-based RAG system
that answers questions about company 10-Q filings using OpenAI models and Chroma.

This document is structured as the questions a reviewer or interviewer is likely to
ask, with precise answers grounded in the actual code (file + line references).

---

## 1. High-level architecture

**Q: Walk me through the architecture.**

A layered RAG pipeline with a clean split between ingestion (offline) and query (online):

- **Ingestion (offline):** [loader.py](src/vectorrag/ingest/loader.py) → [chunker.py](src/vectorrag/ingest/chunker.py) → [embeddings.py](src/vectorrag/embeddings.py) → [vectorstore.py](src/vectorrag/vectorstore.py).
  PDF/HTML/txt → normalized pages → token-bounded overlapping chunks → OpenAI embeddings → Chroma.
- **Query (online):** [rag.py](src/vectorrag/rag.py) orchestrates:
  `sanitize → retrieve → abstain-if-empty → build grounded prompt → LLM → answer + citations`.
- **Entry points:** [cli.py](src/vectorrag/cli.py) (ingest/ask/chat/info) and [api/app.py](api/app.py)
  (FastAPI `/ask`, `/health`), the latter wrapped by Mangum in [lambda_handler.py](api/lambda_handler.py) for Lambda.
- **Cross-cutting:** [config.py](src/vectorrag/config.py) (typed settings), [security.py](src/vectorrag/security.py),
  [tokens.py](src/vectorrag/tokens.py), [memory.py](src/vectorrag/memory.py), [logging_config.py](src/vectorrag/logging_config.py).

**Design principle:** thin entry points, a reusable `RAGEngine`, and dependency injection via
factory functions in [clients.py](src/vectorrag/clients.py) so the engine doesn't construct its own
OpenAI/Chroma clients (testability).

```
                       INGESTION (offline)                          QUERY (online)
 ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌────────┐     ┌──────────────────────────────┐
 │ PDF/HTML│──▶│ loader   │──▶│ chunker    │──▶│embedder│──┐  │  CLI / FastAPI /ask          │
 │  /txt   │   │ (pages)  │   │(token+over-│   │(OpenAI)│  │  └──────────────┬───────────────┘
 └─────────┘   └──────────┘   │ lap chunks)│   └────────┘  │                 ▼
                              └────────────┘               │        sanitize_query
                                                           ▼                 ▼
                                                    ┌──────────────┐   embed query
                                                    │   Chroma     │◀──── retrieve(top_k, min_score)
                                                    │ (cosine,HNSW)│         ▼
                                                    └──────────────┘   token-budget loop
                                                                             ▼
                                                          abstain if empty ──┤
                                                                             ▼
                                                          grounded prompt → ChatLLM (temp 0)
                                                                             ▼
                                                              answer + [S1][S2] citations
```

---

## 2. Chunking

**Q: How do you chunk, and why that way?**

Token-based, not character-based ([chunker.py:31](src/vectorrag/ingest/chunker.py#L31)). Sentences are
split with a regex, then greedily packed up to `chunk_size_tokens` (500). When a chunk fills, the
trailing `chunk_overlap_tokens` (80) worth of sentences seed the next chunk
(`_overlap_tail`, [chunker.py:88](src/vectorrag/ingest/chunker.py#L88)).

- **Why token-based:** chunks fit the embedding model cleanly and the retrieval budget is
  predictable — you can't reason about a token budget if you chunked by characters.
- **Why overlap:** continuity. A fact spanning a chunk boundary would otherwise be unretrievable.
- **Edge case:** a single sentence larger than the chunk size is hard-split on raw token IDs
  (`_hard_split`, [chunker.py:101](src/vectorrag/ingest/chunker.py#L101)) so one giant sentence
  can't blow the budget.
- **Idempotency:** chunk IDs are deterministic — `{source}::p{page}::c{idx}`
  ([chunker.py:83](src/vectorrag/ingest/chunker.py#L83)). Upsert overwrites rather than duplicates,
  so re-ingestion is safe.

---

## 3. Embeddings & vector store

**Q: Which embedding model and why batching?**

`text-embedding-3-small`. Embeddings are batched at 64 ([embeddings.py:16](src/vectorrag/embeddings.py#L16))
to bound request size/cost, with `tenacity` exponential-backoff retry (5 attempts) for transient failures.

**Q: How is similarity computed?**

Chroma collection created with `hnsw:space: cosine` ([vectorstore.py:36](src/vectorrag/vectorstore.py#L36)).
Chroma returns cosine **distance** (0 = identical); converted to similarity with `score = 1.0 - distance`
([vectorstore.py:75](src/vectorrag/vectorstore.py#L75)), then anything below `min_relevance_score` (0.20)
is dropped.

**Key point:** you pass **precomputed embeddings**, so Chroma's built-in embedding function is disabled —
OpenAI is the single source of truth. Otherwise query and document vectors could come from different
models and be incomparable.

**Known limitation:** `1 - distance` assumes a normalized distance range; the 0.20 floor is empirical, not
calibrated. HNSW is approximate (ANN), so recall isn't guaranteed at 100%.

---

## 4. Retrieval & token budget

**Q: What happens at query time, low level?**

[retriever.py:32](src/vectorrag/retriever.py#L32): embed query → Chroma `query(top_k=5, min_score=0.20)` →
then a **token-budget loop**: accumulate snippets while `used_tokens + chunk_tokens <= max_context_tokens`
(6000), breaking once full. Each kept snippet gets a citable id `S1, S2, ...`.

**Why the budget matters:** it's the hard guarantee that the assembled context never overflows the model
window or spikes cost — enforced independently of `top_k`.

---

## 5. Anti-hallucination (the headline feature)

**Q: How do you prevent hallucination?** Four layers, in order:

1. **Abstain on empty retrieval** — if nothing clears the relevance floor, return a fixed
   "I don't have enough information..." and never call the LLM ([rag.py:78](src/vectorrag/rag.py#L78)).
   `grounded=False` is returned so callers can tell.
2. **Grounded system prompt** — "answer ONLY from CONTEXT, say the exact refusal sentence if not present,
   cite every claim, quote figures exactly" ([prompts.py:12](src/vectorrag/prompts.py#L12)).
3. **Temperature 0.0** ([config.py:34](src/vectorrag/config.py#L34)) for determinism/reproducibility.
4. **Mandatory citations** — every snippet carries `[S1]`-style ids and source/page metadata so answers
   are auditable back to a filing page.

**Limitation (own it):** citations are instructed but not *verified* — the system doesn't check that every
`[Sx]` exists or that quoted numbers match the source. A production hardening step is a post-generation
validator.

---

## 6. Security (defense in depth)

**Q: What's your threat model?** ([security.py](src/vectorrag/security.py))

- **DoS/cost:** `MAX_QUERY_CHARS = 4000`, enforced in `sanitize_query` and the Pydantic request model
  ([api/app.py:38](api/app.py#L38)).
- **Prompt injection (two surfaces):** user input *and* retrieved document text are both untrusted.
  `detect_injection` flags known patterns (logged, not blocked); `neutralize_context`
  ([security.py:73](src/vectorrag/security.py#L73)) replaces injection markers in retrieved text with
  `[redacted-instruction]` before it enters the prompt. System-prompt rule #5 is the real defense.
- **Control-char smuggling:** stripped in `sanitize_query` ([security.py:58](src/vectorrag/security.py#L58)).
- **PII/secret redaction before logging:** email/SSN/API-key/card regexes
  ([security.py:32](src/vectorrag/security.py#L32)); queries are redacted *and* truncated to 200 chars
  before logging.

**Limitation:** these are regex heuristics, not guarantees. The strongest control is the grounded prompt.

---

## 7. Memory

**Q: How does conversation memory work and why bound it?**

[memory.py](src/vectorrag/memory.py): a rolling window that drops oldest turns until total tokens fit
`max_history_tokens` (2000) — `_trim` at [memory.py:31](src/vectorrag/memory.py#L31). Caps cost and prevents
unbounded prompt growth.

It is **per-process / per-CLI-session**. The FastAPI `/ask` endpoint deliberately does *not* pass memory
([api/app.py:63](api/app.py#L63)) — the HTTP API is stateless; only `vectorrag chat` is conversational.

**To make the API conversational:** add a session store (DynamoDB/Redis keyed by session id), since Lambda
containers are ephemeral and not sticky.

---

## 8. Config & secrets

**Q: How is configuration managed?**

Pydantic `BaseSettings` with `VECTORRAG_` env prefix and `.env` support
([config.py:17](src/vectorrag/config.py#L17)), with validators (temperature 0–2, overlap < chunk size).

**Secret resolution order** ([config.py:72](src/vectorrag/config.py#L72)): Secrets Manager (prod) →
`OPENAI_API_KEY` env (local). In AWS the key never lives on disk or in an image env var. `_fetch_secret`
is `lru_cache`d ([config.py:89](src/vectorrag/config.py#L89)) so warm Lambda invocations don't re-hit
Secrets Manager.

---

## 9. Deployment / AWS

**Q: How does it run in Lambda?**

Container-image Lambda ([main.tf:32](infra/terraform/main.tf#L32)) with FastAPI behind Mangum
([lambda_handler.py:25](api/lambda_handler.py#L25)). Public **Lambda Function URL** (not API Gateway) —
cheaper/simpler for a single endpoint; swap to API Gateway for WAF/custom domains/throttling
([main.tf:64](infra/terraform/main.tf#L64)).

**The hard problem — "Chroma is a local DB, Lambda is ephemeral. How?"**

The vector DB isn't bundled in the image. On cold start, `hydrate_chroma_from_s3`
([aws_bootstrap.py:19](src/vectorrag/aws_bootstrap.py#L19)) downloads the persisted Chroma files from S3 into
`/tmp/chroma`, no-ops if already hydrated (warm container). Engine + secret are cached per warm container.

**Cold-start cost (own it):** downloading the whole DB into `/tmp` on every cold start doesn't scale past a
modest corpus (`/tmp` is 512MB–10GB, plus download latency). For a larger index, use an **EFS mount** or an
external managed vector DB (called out in [vectorstore.py](src/vectorrag/vectorstore.py#L4)).

---

## 10. Resilience & observability

- **Retries:** tenacity on embeddings (5×) and chat (4×) with exponential backoff; the OpenAI client itself
  is set to `max_retries=0` ([clients.py:17](src/vectorrag/clients.py#L17)) so retry logic lives in one place.
- **Error handling at the edge:** `SecurityError → 400`, any other exception → `502 "Upstream model error"`
  with the real error type logged but not leaked ([api/app.py:64-68](api/app.py#L64-L68)).
- **Structured logging** with token usage logged per completion ([llm.py:37](src/vectorrag/llm.py#L37)) —
  enables cost monitoring.

---

## 11. Known weak spots (own these before they're found)

1. **No reranking** — pure vector top_k; no cross-encoder rerank or hybrid (BM25 + dense) search.
2. **Tables in 10-Qs** — `pypdf.extract_text()` mangles tabular financial data; HTML tables are flattened
   to text ([loader.py:57](src/vectorrag/ingest/loader.py#L57)). The most important data in a 10-Q (the
   financial statements) is the hardest to extract well.
3. **Citation enforcement is prompt-only**, not verified (see §5).
4. **No evaluation harness** — no retrieval recall / answer-faithfulness metrics.
5. **Stateless API has no memory** (§7), and Function URL auth is `NONE`
   ([main.tf:66](infra/terraform/main.tf#L66)) — open endpoint, no per-key throttling.
6. **`min_relevance_score` is a magic number** — not calibrated per corpus.

---

## 12. Quick reference — key tunables

| Setting | Default | Where | Purpose |
|---|---|---|---|
| `chat_model` | `gpt-4o-mini` | [config.py:30](src/vectorrag/config.py#L30) | answer generation |
| `embedding_model` | `text-embedding-3-small` | [config.py:31](src/vectorrag/config.py#L31) | vectorization |
| `temperature` | `0.0` | [config.py:34](src/vectorrag/config.py#L34) | determinism / anti-hallucination |
| `top_k` | `5` | [config.py:38](src/vectorrag/config.py#L38) | candidates retrieved |
| `chunk_size_tokens` | `500` | [config.py:39](src/vectorrag/config.py#L39) | chunk size |
| `chunk_overlap_tokens` | `80` | [config.py:40](src/vectorrag/config.py#L40) | boundary continuity |
| `max_context_tokens` | `6000` | [config.py:41](src/vectorrag/config.py#L41) | context budget cap |
| `min_relevance_score` | `0.20` | [config.py:44](src/vectorrag/config.py#L44) | relevance floor / abstain trigger |
| `max_history_tokens` | `2000` | [config.py:51](src/vectorrag/config.py#L51) | conversation memory cap |
