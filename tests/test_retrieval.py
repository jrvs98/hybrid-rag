from app.domain.schemas import Document
from app.indexes import InMemoryBM25Index, InMemoryDenseIndex
from app.ingestion import chunk_document
from app.retrieval import HybridRetriever, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_deduplicates_and_preserves_branch_scores() -> None:
    chunks = chunk_document(Document(document_id="doc-1", text="alpha beta gamma"))
    bm25_results = [type("Result", (), {"chunk": chunks[0], "score": 2.0})()]
    dense_results = [type("Result", (), {"chunk": chunks[0], "score": 0.9})()]

    results = reciprocal_rank_fusion(
        {"bm25": bm25_results, "dense": dense_results}, rrf_k=1
    )

    assert len(results) == 1
    assert results[0].bm25_score == 2.0
    assert results[0].dense_score == 0.9
    assert results[0].fused_score == 1.0


def test_hybrid_retriever_combines_bm25_and_dense_candidates() -> None:
    chunks = chunk_document(
        Document(document_id="doc-1", text="cats chase mice. dogs guard homes."),
        chunk_size=17,
        overlap=0,
    )
    bm25_index = InMemoryBM25Index()
    dense_index = InMemoryDenseIndex()
    bm25_index.add(chunks)
    dense_index.add(chunks, {chunks[0].chunk_id: [1.0, 0.0], chunks[1].chunk_id: [0.0, 1.0]})

    results = HybridRetriever(bm25_index, dense_index, rrf_k=1).search(
        "dogs homes", [0.1, 1.0], limit=2
    )

    assert [result.chunk.chunk_id for result in results] == [
        chunks[1].chunk_id,
        chunks[0].chunk_id,
    ]
    assert results[0].bm25_score is not None
    assert results[0].dense_score is not None