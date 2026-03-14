# ADR: Why LangChain

## Status

Accepted

## Context

This project loads code from git repositories, splits it into chunks, embeds those chunks, and serves them through a RAG chain backed by a local Ollama model. Choosing a library came down to what already had working integrations for Ollama, Qdrant, and code-aware text splitting.

## Decision

Use LangChain for document loading, text splitting, the Ollama client, and the core `Document` type.

## Rationale

LangChain has maintained integrations for everything this project uses:

- `langchain-ollama` provides `ChatOllama`, which handles connection management, retries, and streaming.
- `langchain-text-splitters` includes `RecursiveCharacterTextSplitter` with language-specific separators (Python, Markdown, C++) and `MarkdownHeaderTextSplitter`.
- `langchain_core.documents.Document` is a simple data class with `page_content` and `metadata` that passes cleanly between ingestion, retrieval, and generation.

The text splitting for code is the part that would be painful to build from scratch. `RecursiveCharacterTextSplitter` handles language-specific edge cases like nested functions, docstrings, and comment blocks that a naive character split gets wrong.

### What we don't use

This project does not use LangChain's higher-level abstractions:

- No `ConversationalRetrievalChain`: conversation memory is a simple list of message dicts managed directly in `RAGChain`.
- No `VectorStoreRetriever` wrapper: `VectorRetriever` and `HybridRetriever` call Qdrant directly via `qdrant-client`.
- No `PromptTemplate` objects: prompts are plain Python f-strings.
- No LangChain agents or tools.

LangChain is a library here, not a framework. The application logic does not go through LangChain's orchestration layer, which keeps it explicit and easy to follow.

### Why not build it ourselves

`ChatOllama` handles connection retries and streaming. `RecursiveCharacterTextSplitter` with language support handles code splitting edge cases that would take time to reproduce. Using `langchain_core.documents.Document` also means the codebase is familiar to other developers who have worked with LangChain.

## Consequences

- Added `langchain-ollama`, `langchain-core`, `langchain-text-splitters` (all pinned to `>=0.3.0`).
- `langchain` itself is at `>=0.3.0`.
- All imports use the modern split-package style (`langchain_core.documents.Document`, not `langchain.schema.Document`).
- Application logic does not depend on LangChain's chain or agent abstractions, so upgrading LangChain packages carries low risk.
