import pytest

from app.domain.schemas import Document
from app.ingestion import chunk_document


def test_chunk_document_preserves_offsets_metadata_and_stable_ids() -> None:
    document = Document(document_id="doc-1", text="abcdefghij", metadata={"team": "search"})

    chunks = chunk_document(document, chunk_size=6, overlap=2)

    assert [chunk.chunk_id for chunk in chunks] == ["doc-1:chunk-0", "doc-1:chunk-1"]
    assert [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        ("abcdef", 0, 6),
        ("efghij", 4, 10),
    ]
    assert chunks[0].metadata == {"team": "search"}


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (5, 5), (5, 6)])
def test_chunk_document_rejects_invalid_window(chunk_size: int, overlap: int) -> None:
    document = Document(document_id="doc-1", text="content")

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=chunk_size, overlap=overlap)