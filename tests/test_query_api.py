from fastapi.testclient import TestClient

from app.domain.schemas import (
    Citation,
    Claim,
    Document,
    GeneratedAnswer,
    VerificationStatus,
)
from app.generation import AnswerGenerator
from app.indexes import InMemoryBM25Index, InMemoryDenseIndex
from app.ingestion import chunk_document
from app.main import app
from app.pipeline import RAGPipeline
from app.retrieval import CrossEncoderReranker, HybridRetriever
from app.verification import CitationVerifier, VerificationAssessment


def build_pipeline() -> RAGPipeline:
    chunks = chunk_document(
        Document(document_id="doc-1", text="The answer is 42."),
        chunk_size=100,
        overlap=0,
    )
    bm25_index = InMemoryBM25Index()
    dense_index = InMemoryDenseIndex()
    bm25_index.add(chunks)
    dense_index.add(chunks, {chunks[0].chunk_id: [1.0, 0.0]})

    def generate(query, evidence):
        return GeneratedAnswer(
            answer="The answer is 42.",
            claims=[Claim(claim_id="claim-1", text="The answer is 42.", citation_ids=["cite-1"])],
            citations=[Citation(citation_id="cite-1", chunk_id=evidence[0].chunk_id)],
        )

    return RAGPipeline(
        retriever=HybridRetriever(bm25_index, dense_index),
        reranker=CrossEncoderReranker(lambda query, text: 1.0),
        generator=AnswerGenerator(generate),
        verifier=CitationVerifier(
            lambda claim, text: VerificationAssessment(
                VerificationStatus.SUPPORTED, 0.99, "The evidence supports the claim."
            )
        ),
    )


def test_query_returns_verified_answer_and_evidence() -> None:
    app.state.query_pipeline = build_pipeline()
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the answer?", "query_vector": [1.0, 0.0]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is True
    assert payload["claims"][0]["citation_ids"] == ["cite-1"]
    assert payload["evidence"][0]["chunk_id"] == "doc-1:chunk-0"


def test_query_returns_service_unavailable_without_pipeline() -> None:
    app.state.query_pipeline = None
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "What is the answer?", "query_vector": [1.0]},
    )

    assert response.status_code == 503