from app.domain.schemas import (
    Citation,
    CitationVerification,
    Claim,
    VerificationStatus,
)


def test_verification_contract_represents_supported_claim() -> None:
    claim = Claim(
        claim_id="claim-1",
        text="The service verifies citations.",
        citation_ids=["citation-1"],
    )
    citation = Citation(citation_id="citation-1", chunk_id="chunk-1")
    verification = CitationVerification(
        claim_id=claim.claim_id,
        citation_id=citation.citation_id,
        status=VerificationStatus.SUPPORTED,
        score=0.98,
        reason="The evidence directly entails the claim.",
    )

    assert verification.status is VerificationStatus.SUPPORTED
    assert claim.citation_ids == [citation.citation_id]
