import math
from dataclasses import dataclass
from typing import Callable

from app.domain.schemas import (
    CitationVerification,
    Evidence,
    GeneratedAnswer,
    VerificationStatus,
    VerifiedAnswer,
)


@dataclass(frozen=True)
class VerificationAssessment:
    status: VerificationStatus
    score: float
    reason: str


class CitationVerifier:
    def __init__(
        self,
        assess: Callable[[str, str], VerificationAssessment],
    ) -> None:
        self._assess = assess

    def verify(
        self,
        answer: GeneratedAnswer,
        evidence: list[Evidence],
    ) -> VerifiedAnswer:
        evidence_by_chunk = {item.chunk_id: item for item in evidence}
        citations_by_id = {citation.citation_id: citation for citation in answer.citations}
        verification: list[CitationVerification] = []

        for claim in answer.claims:
            for citation_id in claim.citation_ids:
                citation = citations_by_id.get(citation_id)
                if citation is None:
                    raise ValueError(f"claim references unknown citation: {citation_id}")
                cited_evidence = evidence_by_chunk.get(citation.chunk_id)
                if cited_evidence is None:
                    raise ValueError(f"citation references unavailable chunk: {citation.chunk_id}")

                assessment = self._assess(claim.text, cited_evidence.text)
                self._validate_assessment(assessment)
                verification.append(
                    CitationVerification(
                        claim_id=claim.claim_id,
                        citation_id=citation.citation_id,
                        status=assessment.status,
                        score=assessment.score,
                        reason=assessment.reason,
                    )
                )

        verified = bool(verification) and all(
            item.status is VerificationStatus.SUPPORTED for item in verification
        )
        return VerifiedAnswer(
            answer=answer.answer,
            claims=answer.claims,
            citations=answer.citations,
            verification=verification,
            verified=verified,
        )

    @staticmethod
    def _validate_assessment(assessment: VerificationAssessment) -> None:
        if not assessment.reason.strip():
            raise ValueError("verification reason must not be empty")
        if not math.isfinite(assessment.score):
            raise ValueError("verification score must be finite")