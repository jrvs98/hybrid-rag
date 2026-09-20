from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: Optional[HttpUrl] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: Optional[HttpUrl] = None
    metadata: dict[str, str] = Field(default_factory=dict)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)


class Evidence(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: Optional[HttpUrl] = None
    page: Optional[int] = Field(default=None, ge=1)
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None
    reranker_score: Optional[float] = None


class Citation(BaseModel):
    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class Claim(BaseModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    UNVERIFIABLE = "unverifiable"


class CitationVerification(BaseModel):
    claim_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    status: VerificationStatus
    score: Optional[float] = None
    reason: str = Field(min_length=1)


class VerifiedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    claims: list[Claim] = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
    verification: list[CitationVerification] = Field(min_length=1)
    verified: bool


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    query_vector: list[float] = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)
    candidate_limit: int = Field(default=50, ge=1, le=500)


class QueryResponse(VerifiedAnswer):
    evidence: list[Evidence] = Field(min_length=1)
