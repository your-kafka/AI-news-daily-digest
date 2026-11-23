from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PoolingMethod = Literal["max", "top3", "combmnz"]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )


class DatabaseSettings(EnvSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "lodestar"
    postgres_user: str = "lodestar"
    postgres_password: SecretStr


class ScraperSettings(EnvSettings):
    youtube_api_key: SecretStr | None = None


class RagSettings(EnvSettings):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size_tokens: int = Field(default=400, gt=0)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    hf_token: SecretStr | None = None
    model_cache_dir: Path = Path.home() / ".cache" / "fastembed"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_query_count: int = Field(default=5, gt=0)
    retrieval_candidates: int = Field(default=100, gt=0, le=1000)
    rrf_k: int = Field(default=60, gt=0)
    pooling_method: PoolingMethod = "max"
    pool_size: int = Field(default=30, gt=0)
    rerank_keep: int = Field(default=20, gt=0)
    rerank_passage_characters: int | None = Field(default=None, gt=0)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)
    retrieval_top_k: int = Field(default=10, gt=0)
    curation_candidates: int = Field(default=15, gt=0)

    @model_validator(mode="after")
    def overlap_fits_in_chunk(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({self.chunk_overlap_tokens}) must be smaller "
                f"than CHUNK_SIZE_TOKENS ({self.chunk_size_tokens})"
            )
        return self


class WorkerSettings(EnvSettings):
    redis_url: str = "redis://localhost:6380/0"
    schedule_timezone: str = "Asia/Kolkata"
    ingest_every_hours: int = Field(default=6, gt=0, le=24)
    ingest_window_hours: float = Field(default=8, gt=0)
    digest_hour: int = Field(default=7, ge=0, le=23)
    digest_window_hours: float = Field(default=24, gt=0)
    profiles_file: Path = PROJECT_ROOT / "config" / "profiles.yaml"
    admin_token: SecretStr | None = None


class GroqSettings(EnvSettings):
    groq_api_key: SecretStr
    groq_model: str


@lru_cache
def database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache
def scraper_settings() -> ScraperSettings:
    return ScraperSettings()


@lru_cache
def rag_settings() -> RagSettings:
    return RagSettings()


@lru_cache
def groq_settings() -> GroqSettings:
    return GroqSettings()


@lru_cache
def worker_settings() -> WorkerSettings:
    return WorkerSettings()
