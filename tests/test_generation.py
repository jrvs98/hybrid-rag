import pytest

from app.domain.schemas import Citation, Claim, Document, GeneratedAnswer
from app.generation import AnswerGenerator, evidence_from_results
from app.ingestion import chunk_document
from app.retrieval import HybridResult


def make_result() -> HybridResult:
    chunk = chunk_document(Document(document_id="doc-1", text="The answer is 42."))[0]
    return HybridResult(chunk=chunk, fused_score=0.8, reranker_score=0.95)


def test_evidence_conversion_preserves_retrieval_provenance() -> None:
    evidence = evidence_from_results([make_result()])

    assert evidence[0].chunk_id == "doc-1:chunk-0"
    assert evidence[0].fused_score == 0.8
    assert evidence[0].reranker_score == 0.95


def test_answer_generator_accepts_only_retrieved_citations() -> None:
    def generate(query, evidence):
        return GeneratedAnswer(
            answer="The answer is 42.",
            claims=[Claim(claim_id="claim-1", text="The answer is 42.", citation_ids=["cite-1"])],
            citations=[Citation(citation_id="cite-1", chunk_id=evidence[0].chunk_id)],
        )

    answer, evidence = AnswerGenerator(generate).generate("What is the answer?", [make_result()])

    assert answer.citations[0].chunk_id == evidence[0].chunk_id


def test_answer_generator_rejects_unknown_citation_chunk() -> None:
    def generate(query, evidence):
        return GeneratedAnswer(
            answer="Unsupported.",
            claims=[Claim(claim_id="claim-1", text="Unsupported.", citation_ids=["cite-1"])],
            citations=[Citation(citation_id="cite-1", chunk_id="missing")],
        )

    with pytest.raises(ValueError, match="unavailable chunk"):
        AnswerGenerator(generate).generate("Question", [make_result()])


def test_answer_generator_rejects_unknown_claim_citation() -> None:
    def generate(query, evidence):
        return GeneratedAnswer(
            answer="The answer is 42.",
            claims=[Claim(claim_id="claim-1", text="The answer is 42.", citation_ids=["missing"])],
            citations=[],
        )

    with pytest.raises(ValueError, match="too_short"):
        AnswerGenerator(generate).generate("Question", [make_result()])