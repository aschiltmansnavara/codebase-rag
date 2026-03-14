#!/bin/bash
# Entrypoint script for the Codebase RAG Docker container.
# Auto-ingestion of the default repo is handled by the Streamlit UI
# (see components.py) so the app is available immediately on startup.
set -e

# Wait for Qdrant to be ready
QDRANT_URL="http://${QDRANT_HOST:-localhost}:${QDRANT_PORT:-6333}"
echo "Waiting for Qdrant at ${QDRANT_URL}..."
for i in $(seq 1 30); do
    if curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    sleep 2
done

# Pull default model if needed
MODEL="${LLM_MODEL_NAME:-sam860/LFM2:350m}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "Ensuring model '${MODEL}' is available in Ollama..."
curl -sf "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${MODEL}\"}" > /dev/null 2>&1 || echo "Warning: Could not pull model ${MODEL}"

# Start Streamlit
exec .venv/bin/streamlit run src/codebase_rag/app/main.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
