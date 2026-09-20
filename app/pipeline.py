from app.domain.schemas import QueryRequest, QueryResponse
from app.generation import AnswerGenerator
from app.retrieval import CrossEncoderReranker, HybridRetriever
from app.verification import CitationVerifier


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: AnswerGenerator,
        verifier: CitationVerifier,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._verifier = verifier

    def query(self, request: QueryRequest) -> QueryResponse:
        retrieved = self._retriever.search(
            request.query,
            request.query_vector,
            limit=request.candidate_limit,
            candidate_limit=request.candidate_limit,
        )
        reranked = self._reranker.rerank(request.query, retrieved, limit=request.limit)
        generated, evidence = self._generator.generate(request.query, reranked)
        verified = self._verifier.verify(generated, evidence)
        return QueryResponse(
            answer=verified.answer,
            claims=verified.claims,
            citations=verified.citations,
            verification=verified.verification,
            verified=verified.verified,
            evidence=evidence,
        )