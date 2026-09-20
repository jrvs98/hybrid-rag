import pytest

from app.domain.schemas import Document
from app.ingestion import chunk_document
from app.retrieval import CrossEncoderReranker, HybridResult


def test_cross_encoder_reranker_reorders_candidates_and_preserves_scores() -> None:
    chunks = chunk_document(
        Document(document_id="doc-1", text="alpha evidence. beta evidence."),
        chunk_size=16,
        overlap=0,
    )
    candidates = [
        HybridResult(
            chunk=chunks[0], fused_score=0.9, bm25_score=1.2, dense_score=0.4
        ),
        HybridResult(
            chunk=chunks[1], fused_score=0.8, bm25_score=0.5, dense_score=0.7
        ),
    ]

    reranked = CrossEncoderReranker(
        lambda query, text: 0.95 if "beta" in text else 0.2
    ).rerank("query", candidates)

    assert reranked[0].chunk.chunk_id == chunks[1].chunk_id
    assert reranked[0].reranker_score == 0.95
    assert reranked[0].fused_score == 0.8
    assert reranked[0].bm25_score == 0.5
    assert reranked[0].dense_score == 0.7


def test_cross_encoder_reranker_limits_results() -> None:
    chunks = chunk_document(Document(document_id="doc-1", text="alpha beta"))
    candidate = HybridResult(chunk=chunks[0], fused_score=1.0)

    assert (
        len(CrossEncoderReranker(lambda query, text: 1.0).rerank("query", [candidate], limit=1))
        == 1
    )
    assert CrossEncoderReranker(lambda query, text: 1.0).rerank("query", [candidate], limit=0) == []


def test_cross_encoder_reranker_rejects_non_finite_scores() -> None:
    chunks = chunk_document(Document(document_id="doc-1", text="alpha"))
    candidate = HybridResult(chunk=chunks[0], fused_score=1.0)

    with pytest.raises(ValueError, match="finite"):
        CrossEncoderReranker(lambda query, text: float("nan")).rerank("query", [candidate])