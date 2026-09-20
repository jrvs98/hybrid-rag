from dataclasses import dataclass
from typing import Callable, Optional

from app.domain.schemas import DocumentChunk
from app.indexes import IndexedChunk, InMemoryBM25Index, InMemoryDenseIndex


@dataclass(frozen=True)
class HybridResult:
    chunk: DocumentChunk
    fused_score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    reranker_score: Optional[float] = None


def reciprocal_rank_fusion(
    ranked_results: dict[str, list[IndexedChunk]],
    *,
    weights: Optional[dict[str, float]] = None,
    rrf_k: int = 60,
    limit: int = 10,
) -> list[HybridResult]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if limit < 1:
        return []

    source_weights = weights or {source: 1.0 for source in ranked_results}
    if any(weight < 0 for weight in source_weights.values()):
        raise ValueError("fusion weights must be non-negative")

    by_chunk_id: dict[str, HybridResult] = {}
    scores: dict[str, float] = {}
    for source, results in ranked_results.items():
        weight = source_weights.get(source, 0.0)
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
            existing = by_chunk_id.get(chunk_id)
            if existing is None:
                existing = HybridResult(chunk=result.chunk, fused_score=0.0)
                by_chunk_id[chunk_id] = existing
            if source == "bm25":
                by_chunk_id[chunk_id] = HybridResult(
                    chunk=result.chunk,
                    fused_score=0.0,
                    bm25_score=result.score,
                    dense_score=existing.dense_score,
                )
            elif source == "dense":
                by_chunk_id[chunk_id] = HybridResult(
                    chunk=result.chunk,
                    fused_score=0.0,
                    bm25_score=existing.bm25_score,
                    dense_score=result.score,
                )

    ranked = sorted(
        by_chunk_id,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )
    return [
        HybridResult(
            chunk=by_chunk_id[chunk_id].chunk,
            fused_score=scores[chunk_id],
            bm25_score=by_chunk_id[chunk_id].bm25_score,
            dense_score=by_chunk_id[chunk_id].dense_score,
        )
        for chunk_id in ranked[:limit]
    ]


class CrossEncoderReranker:
    def __init__(self, score: Callable[[str, str], float]) -> None:
        self._score = score

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        *,
        limit: Optional[int] = None,
    ) -> list[HybridResult]:
        if limit is not None and limit < 1:
            return []

        reranked: list[HybridResult] = []
        for candidate in candidates:
            reranker_score = float(self._score(query, candidate.chunk.text))
            if reranker_score != reranker_score or reranker_score in (float("inf"), float("-inf")):
                raise ValueError("reranker score must be finite")
            reranked.append(
                HybridResult(
                    chunk=candidate.chunk,
                    fused_score=candidate.fused_score,
                    bm25_score=candidate.bm25_score,
                    dense_score=candidate.dense_score,
                    reranker_score=reranker_score,
                )
            )
        reranked.sort(
            key=lambda result: (-result.reranker_score, -result.fused_score, result.chunk.chunk_id)
        )
        return reranked if limit is None else reranked[:limit]


class HybridRetriever:
    def __init__(
        self,
        bm25_index: InMemoryBM25Index,
        dense_index: InMemoryDenseIndex,
        *,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if bm25_weight < 0 or dense_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        self._bm25_index = bm25_index
        self._dense_index = dense_index
        self._rrf_k = rrf_k
        self._weights = {"bm25": bm25_weight, "dense": dense_weight}

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        candidate_limit: int = 50,
    ) -> list[HybridResult]:
        if candidate_limit < 1:
            return []
        return reciprocal_rank_fusion(
            {
                "bm25": self._bm25_index.search(query, limit=candidate_limit),
                "dense": self._dense_index.search(query_vector, limit=candidate_limit),
            },
            weights=self._weights,
            rrf_k=self._rrf_k,
            limit=limit,
        )