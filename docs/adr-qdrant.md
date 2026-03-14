# ADR: Why Qdrant over Milvus/MinIO

## Status

Accepted

## Context

The original system used Milvus as its vector database and MinIO for object storage (chat history). This combination required three Docker containers (Milvus, MinIO, etcd) with complex networking, volume management, and health-check coordination. The development experience was poor:

- Milvus startup was slow and fragile. Services frequently failed health checks on cold starts.
- MinIO required separate bucket initialization scripts and IAM management.
- The etcd dependency added another failure point with no direct value to the application.
- Debugging vector storage issues required understanding the Milvus/etcd/MinIO interaction model.
- Total memory footprint of the stack exceeded 2 GB at idle.

## Decision

Replace Milvus + MinIO with:

1. **Qdrant** for vector storage (single container, ~200 MB RAM).
2. **SQLite** for chat history persistence (file-based, zero infrastructure).

## Rationale

### Qdrant advantages

- **Single container.** Qdrant runs as one process with embedded storage. No etcd, no MinIO, no multi-service coordination.
- **Native sparse vector support.** Qdrant supports sparse vectors natively, which could be used for BM25 in future. Milvus required external workarounds.
- **Built-in dashboard.** The Qdrant web UI at `:6333/dashboard` provides collection inspection, point browsing, and search testing without additional tools. Milvus required Attu (yet another container).
- **Simpler client API.** `qdrant-client` provides a straightforward Python API. The `pymilvus` API had frequent breaking changes across versions and required schema definition boilerplate.
- **Payload filtering.** Qdrant supports filtering by metadata payloads (used for listing ingested repositories via the `facet()` API), which maps cleanly to the application's needs.
- **Deterministic point IDs.** Qdrant supports UUID-based point IDs, enabling idempotent upserts. Chunks are assigned deterministic IDs based on `source_path + chunk_index` via UUID5, so re-ingesting the same content overwrites existing points without creating duplicates.

### SQLite for chat history

MinIO was being used as a key-value store for serialised chat objects, which is a misuse of an object storage system. SQLite is a better fit:

- Zero infrastructure: no container, no network, no credentials.
- ACID transactions for chat persistence.
- Direct SQL queries for debugging and inspection.
- The `sqlite3` CLI is available everywhere.

### What we gave up

- Milvus has more mature support for very large-scale deployments (billions of vectors). At the scale of a few repositories (~50k vectors), this is irrelevant.
- Milvus has GPU-accelerated indexing. Not needed at this scale.

## Consequences

- Docker Compose went from 5+ services to 5 (Qdrant, Ollama, Langfuse, Postgres, app).
- Cold-start time dropped from ~60 seconds to ~10 seconds.
- Memory footprint at idle dropped from ~2 GB to ~500 MB.
- The `VectorStoreProtocol` abstraction was introduced, making it possible to swap vector backends in future without touching the retrieval layer.
- Chat history is now a single file (`data/chat_history.db`) that can be backed up, inspected, or deleted trivially.
