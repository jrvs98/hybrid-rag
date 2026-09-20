from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


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
