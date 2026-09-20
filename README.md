# Hybrid RAG

API-first hybrid-search RAG service with dense retrieval, BM25, cross-encoder reranking, and claim-level citation verification.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive documentation is at `/docs`.

## Current foundation

- FastAPI application with health and readiness endpoints
- Provider-neutral settings for embeddings, generation, reranking, and verification
- Domain schemas for evidence, claims, citations, and verification results
- Unit tests for the foundation contract
