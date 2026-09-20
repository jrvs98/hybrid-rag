from app.domain.schemas import Document
from app.indexes import InMemoryBM25Index, InMemoryDenseIndex
from app.ingestion import chunk_document


def make_chunks():
    return chunk_document(
        Document(document_id="doc-1", text="cats chase mice. dogs guard homes."),
        chunk_size=17,
        overlap=0,
    )


def test_bm25_ranks_matching_chunk_first() -> None:
    index = InMemoryBM25Index()
    chunks = make_chunks()
    index.add(chunks)

    results = index.search("dogs homes")

    assert results
    assert results[0].chunk.text == "dogs guard homes."
    assert results[0].score > 0


def test_dense_index_ranks_by_cosine_similarity() -> None:
    index = InMemoryDenseIndex()
    chunks = make_chunks()
    index.add(chunks, {chunks[0].chunk_id: [1.0, 0.0], chunks[1].chunk_id: [0.0, 1.0]})

    results = index.search([0.1, 1.0])

    assert results[0].chunk.chunk_id == chunks[1].chunk_id
    assert results[0].score > results[1].score