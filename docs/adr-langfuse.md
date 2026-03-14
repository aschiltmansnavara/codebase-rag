# ADR: Why Langfuse

## Status

Accepted

## Context

RAG systems are hard to debug. When an answer is wrong, the root cause can be in any of three places:

1. **Retrieval:** the relevant documents weren't in the top-k results.
2. **Prompt:** the context was retrieved but the prompt didn't guide the LLM to use it.
3. **Generation:** the LLM had the right context but generated a poor answer anyway.

Without observability, debugging requires adding print statements, re-running queries, and guessing. A tracing solution that records the full pipeline (retrieval results, prompt text, generation output, latencies) makes debugging systematic instead of guesswork.

## Decision

Integrate Langfuse as an optional LLM observability layer. Tracing is off by default and enabled via environment variables.

## How it is used

### Integration points

Langfuse is integrated in `src/codebase_rag/llm/rag_chain.py` via the Langfuse Python SDK:

1. **Trace per query.** Each call to `RAGChain.run()` creates a Langfuse trace with the user query as input.
2. **Retrieval span.** A child span records how many documents were retrieved, the retrieval latency, and which documents were returned.
3. **Generation span.** A child span records the prompt length, the generated answer length, and generation latency.
4. **Trace output.** The complete result (answer, sources, metrics) is attached to the trace.

Every question asked through the UI or eval framework shows up in Langfuse.

### What you can see in Langfuse

- Retrieval quality per query (which documents, relevance, latency)
- The exact prompt sent to the LLM, including conversation history
- Latency breakdown (total, retrieval, generation)
- Evaluation traces from `evals/run_eval.py` correlated with retrieval scores

### Configuration

Langfuse is disabled by default and requires explicit opt-in:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

The Docker Compose configuration includes a Langfuse instance (with its Postgres dependency) that starts automatically. To obtain API keys, visit http://localhost:3000 after starting the stack and create a project.

### Lazy initialization

The Langfuse client is initialized lazily on first use (singleton pattern in `_get_langfuse()`). If Langfuse is unreachable or credentials are invalid, the application continues to function normally. Tracing failures are logged as warnings but do not affect the RAG pipeline.

## Rationale

### Why Langfuse over alternatives

| Alternative | Why not |
|-------------|---------|
| **LangSmith** | Requires a LangChain Cloud account. This project runs entirely locally with no cloud dependencies. |
| **OpenTelemetry + Jaeger** | General-purpose tracing. Powerful, but requires significant instrumentation work for LLM-specific concerns (prompt text, token counts, retrieval documents). |
| **Weights & Biases** | Cloud-first with a heavier SDK. Overkill for a local development project. |
| **Print statements / logging** | Already used for basic debugging, but doesn't provide a queryable UI, trace correlation, or historical comparison. |

Langfuse was chosen because:
- It's open-source and self-hostable (single Docker container + Postgres).
- Its Python SDK is lightweight and non-intrusive.
- Its web UI is purpose-built for LLM trace inspection.
- It integrates well with evaluation frameworks (traces from `run_eval.py` are directly visible).

## Consequences

- Added `langfuse >=2.0.0,<3` to dependencies.
- Added Langfuse + Postgres services to `docker/compose-dev.yml`.
- The RAG chain has ~20 lines of optional tracing code (guarded by `if trace:` checks).
- When disabled, the only cost is one environment variable check per `RAGChain` initialization.
- Evaluation results from `evals/run_eval.py` are automatically traced when Langfuse is enabled, providing a persistent record of evaluation runs.
