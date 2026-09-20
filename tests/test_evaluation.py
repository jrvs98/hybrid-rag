import pytest

from app.domain.schemas import (
    Citation,
    CitationVerification,
    Claim,
    Document,
    VerificationStatus,
    VerifiedAnswer,
)
from app.evaluation import evaluate_citations, evaluate_retrieval, mean_metric
from app.ingestion import chunk_document
from app.retrieval import HybridResult


def test_evaluate_retrieval_calculates_recall_and_ndcg() -> None:
    chunks = chunk_document(
        Document(document_id="doc-1", text="one two three"), chunk_size=4, overlap=0
    )
    results = [
        HybridResult(chunk=chunks[1], fused_score=0.9),
        HybridResult(chunk=chunks[0], fused_score=0.8),
        HybridResult(chunk=chunks[2], fused_score=0.7),
    ]

    metrics = evaluate_retrieval(results, {chunks[0].chunk_id, chunks[1].chunk_id}, k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.ndcg_at_k == 1.0


def test_evaluate_citations_measures_pair_support_and_unsupported_claims() -> None:
    answer = VerifiedAnswer(
        answer="Answer.",
        claims=[
            Claim(claim_id="claim-1", text="Supported.", citation_ids=["cite-1"]),
            Claim(claim_id="claim-2", text="Unsupported.", citation_ids=["cite-2"]),
        ],
        citations=[
            Citation(citation_id="cite-1", chunk_id="chunk-1"),
            Citation(citation_id="cite-2", chunk_id="chunk-2"),
        ],
        verification=[
            CitationVerification(
                claim_id="claim-1",
                citation_id="cite-1",
                status=VerificationStatus.SUPPORTED,
                score=0.9,
                reason="Supported.",
            ),
            CitationVerification(
                claim_id="claim-2",
                citation_id="cite-2",
                status=VerificationStatus.INSUFFICIENT,
                score=0.2,
                reason="Insufficient.",
            ),
        ],
        verified=False,
    )

    metrics = evaluate_citations(answer)

    assert metrics.support_rate == 0.5
    assert metrics.unsupported_claim_rate == 0.5


def test_evaluation_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval([], {"chunk-1"}, k=0)
    with pytest.raises(ValueError):
        mean_metric([])