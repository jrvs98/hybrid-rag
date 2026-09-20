from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "hybrid-rag"
    environment: str = "development"
    log_level: str = "INFO"
    embedding_provider: str = "local"
    generation_provider: str = "local"
    reranker_provider: str = "local"
    verifier_provider: str = "local"
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag_chunks"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
