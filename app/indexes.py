import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.domain.schemas import DocumentChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class IndexedChunk:
    chunk: DocumentChunk
    score: float


class InMemoryBM25Index:
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def add(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            terms = Counter(tokenize(chunk.text))
            self._chunks[chunk.chunk_id] = chunk
            self._term_frequencies[chunk.chunk_id] = terms
            self._document_frequency.update(terms.keys())
        total_terms = sum(
            sum(frequencies.values()) for frequencies in self._term_frequencies.values()
        )
        self._average_length = total_terms / len(self._chunks) if self._chunks else 0.0

    def search(self, query: str, *, limit: int = 10) -> list[IndexedChunk]:
        if limit < 1:
            return []
        query_terms = tokenize(query)
        if not query_terms or not self._chunks:
            return []

        document_count = len(self._chunks)
        scores: dict[str, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        for term in query_terms:
            document_frequency = self._document_frequency.get(term, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for chunk_id, frequencies in self._term_frequencies.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                length = sum(frequencies.values())
                normalization = 1 - b + b * length / self._average_length
                scores[chunk_id] += inverse_frequency * (term_frequency * (k1 + 1)) / (
                    term_frequency + k1 * normalization
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [IndexedChunk(self._chunks[chunk_id], score) for chunk_id, score in ranked[:limit]]


class InMemoryDenseIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._chunks: dict[str, DocumentChunk] = {}

    def add(self, chunks: list[DocumentChunk], vectors: dict[str, list[float]]) -> None:
        for chunk in chunks:
            vector = vectors.get(chunk.chunk_id)
            if vector is None:
                raise ValueError(f"missing vector for chunk_id: {chunk.chunk_id}")
            if not vector or not any(vector):
                raise ValueError(f"vector must be non-zero for chunk_id: {chunk.chunk_id}")
            if chunk.chunk_id in self._chunks:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = tuple(vector)

    def search(self, query_vector: list[float], *, limit: int = 10) -> list[IndexedChunk]:
        if limit < 1 or not query_vector or not any(query_vector):
            return []
        query = tuple(query_vector)
        query_norm = math.sqrt(sum(value * value for value in query))
        ranked: list[IndexedChunk] = []
        for chunk_id, vector in self._vectors.items():
            if len(vector) != len(query):
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = sum(left * right for left, right in zip(query, vector)) / (
                query_norm * vector_norm
            )
            ranked.append(IndexedChunk(self._chunks[chunk_id], score))
        ranked.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return ranked[:limit]