import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.domain.schemas import VerificationStatus, VerifiedAnswer
from app.retrieval import HybridResult


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True)
class CitationMetrics:
    support_rate: float
    unsupported_claim_rate: float


def evaluate_retrieval(
    results: list[HybridResult],
    relevant_chunk_ids: set[str],
    *,
    k: int = 10,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    ranked_ids = [result.chunk.chunk_id for result in results[:k]]
    hits = sum(chunk_id in relevant_chunk_ids for chunk_id in ranked_ids)
    recall = hits / len(relevant_chunk_ids)

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids, start=1)
        if chunk_id in relevant_chunk_ids
    )
    ideal_hits = min(k, len(relevant_chunk_ids))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return RetrievalMetrics(recall_at_k=recall, ndcg_at_k=dcg / ideal_dcg)


def evaluate_citations(answer: VerifiedAnswer) -> CitationMetrics:
    pair_results = answer.verification
    if not pair_results:
        raise ValueError("answer must contain verification results")

    supported = sum(item.status is VerificationStatus.SUPPORTED for item in pair_results)
    claims_with_unsupported = {
        item.claim_id for item in pair_results if item.status is not VerificationStatus.SUPPORTED
    }
    return CitationMetrics(
        support_rate=supported / len(pair_results),
        unsupported_claim_rate=len(claims_with_unsupported) / len(answer.claims),
    )


def mean_metric(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)