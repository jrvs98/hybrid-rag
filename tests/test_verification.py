import pytest

from app.domain.schemas import Citation, Claim, Document, GeneratedAnswer, VerificationStatus
from app.generation import evidence_from_results
from app.ingestion import chunk_document
from app.retrieval import HybridResult
from app.verification import CitationVerifier, VerificationAssessment


def make_answer() -> tuple[GeneratedAnswer, list]:
    chunk = chunk_document(
        Document(document_id="doc-1", text="The answer is 42.")
    )[0]
    evidence = evidence_from_results([HybridResult(chunk=chunk, fused_score=0.8)])
    answer = GeneratedAnswer(
        answer="The answer is 42.",
        claims=[Claim(claim_id="claim-1", text="The answer is 42.", citation_ids=["cite-1"])],
        citations=[Citation(citation_id="cite-1", chunk_id=chunk.chunk_id)],
    )
    return answer, evidence


def test_citation_verifier_marks_supported_claim_verified() -> None:
    answer, evidence = make_answer()

    result = CitationVerifier(
        lambda claim, text: VerificationAssessment(
            VerificationStatus.SUPPORTED, 0.98, "The evidence entails the claim."
        )
    ).verify(answer, evidence)

    assert result.verified is True
    assert result.verification[0].status is VerificationStatus.SUPPORTED


@pytest.mark.parametrize(
    "status",
    [
        VerificationStatus.CONTRADICTED,
        VerificationStatus.INSUFFICIENT,
        VerificationStatus.UNVERIFIABLE,
    ],
)
def test_non_supported_pair_prevents_verified_answer(status: VerificationStatus) -> None:
    answer, evidence = make_answer()

    result = CitationVerifier(
        lambda claim, text: VerificationAssessment(
            status, 0.2, "The evidence does not support the claim."
        )
    ).verify(answer, evidence)

    assert result.verified is False
    assert result.verification[0].status is status


def test_citation_verifier_rejects_unknown_evidence() -> None:
    answer, evidence = make_answer()
    answer = answer.model_copy(
        update={"citations": [Citation(citation_id="cite-1", chunk_id="missing")]}
    )

    with pytest.raises(ValueError, match="unavailable chunk"):
        verifier = CitationVerifier(
            lambda claim, text: VerificationAssessment(
                VerificationStatus.SUPPORTED, 1.0, "ok"
            )
        )
        verifier.verify(answer, evidence)