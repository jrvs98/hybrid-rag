from typing import Callable

from app.domain.schemas import Citation, Claim, Evidence, GeneratedAnswer
from app.retrieval import HybridResult


def evidence_from_results(results: list[HybridResult]) -> list[Evidence]:
    return [
        Evidence(
            chunk_id=result.chunk.chunk_id,
            document_id=result.chunk.document_id,
            text=result.chunk.text,
            source_uri=result.chunk.source_uri,
            start_offset=result.chunk.start_offset,
            end_offset=result.chunk.end_offset,
            fused_score=result.fused_score,
            dense_score=result.dense_score,
            bm25_score=result.bm25_score,
            reranker_score=result.reranker_score,
        )
        for result in results
    ]


class AnswerGenerator:
    def __init__(
        self,
        generate: Callable[[str, list[Evidence]], GeneratedAnswer],
    ) -> None:
        self._generate = generate

    def generate(
        self,
        query: str,
        results: list[HybridResult],
    ) -> tuple[GeneratedAnswer, list[Evidence]]:
        evidence = evidence_from_results(results)
        answer = self._generate(query, evidence)
        self._validate_citations(answer, evidence)
        return answer, evidence

    @staticmethod
    def _validate_citations(answer: GeneratedAnswer, evidence: list[Evidence]) -> None:
        evidence_by_chunk = {item.chunk_id: item for item in evidence}
        citations_by_id: dict[str, Citation] = {}
        claims_by_id: dict[str, Claim] = {}

        for citation in answer.citations:
            if citation.citation_id in citations_by_id:
                raise ValueError(f"duplicate citation_id: {citation.citation_id}")
            if citation.chunk_id not in evidence_by_chunk:
                raise ValueError(f"citation references unavailable chunk: {citation.chunk_id}")
            citations_by_id[citation.citation_id] = citation

        for claim in answer.claims:
            if claim.claim_id in claims_by_id:
                raise ValueError(f"duplicate claim_id: {claim.claim_id}")
            if any(citation_id not in citations_by_id for citation_id in claim.citation_ids):
                raise ValueError(f"claim has an unknown citation: {claim.claim_id}")
            claims_by_id[claim.claim_id] = claim