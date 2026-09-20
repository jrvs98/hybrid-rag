from app.domain.schemas import Document, DocumentChunk


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_number = 0
    while start < len(document.text):
        end = min(start + chunk_size, len(document.text))
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:chunk-{chunk_number}",
                document_id=document.document_id,
                text=document.text[start:end],
                source_uri=document.source_uri,
                metadata=document.metadata,
                start_offset=start,
                end_offset=end,
            )
        )
        if end == len(document.text):
            break
        start = end - overlap
        chunk_number += 1
    return chunks