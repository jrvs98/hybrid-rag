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

Run validation with:

```bash
pytest
ruff check app tests
```

## Pipeline

The service is organized as an evidence-traceable pipeline:

```text
documents -> chunking -> BM25 + dense indexes -> hybrid retrieval
					-> cross-encoder reranking -> answer generation
					-> claim-level citation verification
```

The current implementation includes:

- Stable, metadata-preserving document chunks with source offsets
- In-memory BM25 and dense cosine-similarity indexes
- Reciprocal Rank Fusion for hybrid retrieval
- Injectable cross-encoder reranking
- Structured answers with claim and citation IDs
- Claim-level verification with supported, contradicted, insufficient, and unverifiable statuses
- Recall@K, nDCG@K, citation support, and unsupported-claim evaluation metrics

## API

Available endpoints:

- `GET /health` reports service health.
- `GET /ready` reports readiness.
- `POST /query` runs the complete retrieval and verification pipeline.

Example query request:

```json
{
	"query": "What is the answer?",
	"query_vector": [0.12, -0.04, 0.91],
	"limit": 5,
	"candidate_limit": 50
}
```

The query response contains the answer, claims, citations, verification results, and retrieved evidence. Each response includes an `X-Request-ID` header; clients may provide their own request ID.

The application exposes the endpoint contract, but the default app instance does not create provider or index dependencies. Until `app.state.query_pipeline` is configured, `POST /query` returns `503`.

## Configuration

Copy `.env.example` to `.env` and set provider and storage settings as needed. `CORS_ALLOWED_ORIGINS` accepts a comma-separated list and is empty by default.
