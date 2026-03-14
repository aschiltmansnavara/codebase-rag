# ADR: Document Chunking Strategy

## Status

Accepted

## Context

When ingesting codebases for RAG, documents must be split into chunks small enough for embedding models to process effectively, yet large enough to retain meaningful context. The chunking strategy affects retrieval quality: chunks that are too small lose context, chunks that are too large dilute the signal.

## Decision

### Chunk size: 1000 characters with 200-character overlap

- **1000 characters** is a common default for code RAG systems. It's large enough to capture a complete function or class definition in most cases, while small enough to produce focused embeddings.
- **200-character overlap** (20%) so context at chunk boundaries isn't lost. A function call that appears at the end of one chunk will also appear at the start of the next, preventing retrieval failures for queries about boundary content.
- These values align with the typical context window of the embedding model (`all-mpnet-base-v2`, max 384 tokens ≈ ~1500 characters), keeping chunks well within the model's effective range.

### Language-specific splitting

Three strategies are used based on file type:

1. **Python files** (`.py`, `.ipynb`) → `RecursiveCharacterTextSplitter.from_language(language="python")`
   - Splits on Python-specific boundaries: class definitions, function definitions, decorators, blank lines between top-level blocks.
   - Preserves complete logical units (e.g., a full method) whenever possible.
   - Prevents splitting mid-expression or mid-docstring.

2. **Markdown/RST files** (`.md`, `.rst`) → `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
   - First pass: splits by headers (`#`, `##`, `###`, etc.) to capture document hierarchy.
   - Header metadata is preserved in chunk metadata (e.g., `header_1: "API Reference"`, `header_2: "Authentication"`).
   - Second pass: further splits oversized sections using the recursive splitter with markdown-aware separators.
   - This two-pass approach ensures that retrieval results include the section hierarchy, which helps when answering questions about doc structure.

3. **All other files** → `RecursiveCharacterTextSplitter` with default separators
   - Falls back to splitting on `\n\n`, `\n`, ` `, then character-level as a last resort.
   - Suitable for config files, YAML, TOML, plain text, etc.

### Content hashing

Each chunk includes a SHA-256 hash of its content in metadata (`content_hash`). This enables:
- Detecting whether a chunk has changed between ingestion runs.
- Future deduplication or change-tracking optimizations.

### Deterministic chunk IDs

Chunk point IDs in the vector store are deterministic, derived from `source_path + chunk_index` via UUID5. This means:
- Re-ingesting the same file overwrites existing chunks in place (idempotent).
- No duplicate chunks accumulate across runs.

## Consequences

- The chunking parameters are tuned for Python codebases and English documentation. Other languages (Java, Rust, etc.) would benefit from adding language-specific splitters.
- The 1000/200 split works well for the `all-mpnet-base-v2` embedding model. If the embedding model is changed to one with a significantly different context window, these values should be re-evaluated.
- Markdown header preservation means markdown chunks carry richer metadata than code chunks, which can improve retrieval for documentation-heavy repos.
