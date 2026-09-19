"""
Centralised settings — loaded once from environment / .env file.
All components import from here; nothing reads os.environ directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────
    LLM_PROVIDER: str = "google"
    # Supported values: google | openai | anthropic | groq | ollama

    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Ollama — local / India-hosted open model swap-in point
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # Optional hard override of model name for any provider.
    # Defaults per provider: google→gemini-1.5-flash, openai→gpt-4o-mini,
    # anthropic→claude-3-haiku-20240307, groq→llama3-8b-8192
    LLM_MODEL_OVERRIDE: str = ""

    @property
    def llm_model_name(self) -> str:
        if self.LLM_MODEL_OVERRIDE:
            return self.LLM_MODEL_OVERRIDE
        defaults = {
            "google": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "groq": "qwen/qwen3.8-27b",
            "ollama": self.OLLAMA_MODEL,
        }
        return defaults.get(self.LLM_PROVIDER, "gemini-2.5-flash")

    # ── Embeddings ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers"
    # sentence-transformers | openai
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    # Dimension for all-MiniLM-L6-v2 = 384; text-embedding-3-small = 1536
    EMBEDDING_DIM: int = 384

    # ── PostgreSQL ────────────────────────────────────────────────
    POSTGRES_URL: str = (
        "postgresql+asyncpg://bis:bispassword@localhost:5432/bis_standards"
    )
    POSTGRES_SYNC_URL: str = (
        "postgresql://bis:bispassword@localhost:5432/bis_standards"
    )

    # ── Neo4j ─────────────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "bispassword"

    # ── Qdrant ────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_STANDARDS: str = "standards_vectors"
    QDRANT_COLLECTION_FULLTEXT: str = "standards_fulltext_chunks"

    # ── Data paths ────────────────────────────────────────────────
    DATA_DIR: Path = Path("BIS_Sahayak_Clean_Data/clean")

    @property
    def standards_json(self) -> Path:
        return self.DATA_DIR / "standards_clean.json"

    @property
    def whitelist_json(self) -> Path:
        return self.DATA_DIR / "is_code_whitelist_clean.json"

    @property
    def relationships_json(self) -> Path:
        return self.DATA_DIR / "standards" / "standards_relationships.json"

    @property
    def qco_json(self) -> Path:
        return self.DATA_DIR / "regulatory" / "qco_orders.json"

    @property
    def certification_schemes_json(self) -> Path:
        return self.DATA_DIR / "regulatory" / "certification_schemes.json"

    @property
    def product_rules_json(self) -> Path:
        return self.DATA_DIR / "regulatory" / "product_rules.json"

    @property
    def thesaurus_json(self) -> Path:
        return self.DATA_DIR / "knowledge" / "procurement_thesaurus.json"

    @property
    def category_map_json(self) -> Path:
        return self.DATA_DIR / "knowledge" / "category_keyword_maps.json"

    @property
    def eval_queries_json(self) -> Path:
        return self.DATA_DIR / "evaluation" / "queries_master.json"

    # ── API ───────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # ── Retrieval tuning ──────────────────────────────────────────
    RETRIEVE_TOP_K: int = 10
    GRAPH_MAX_HOPS: int = 2

    # ── Chunking (for full_text embedding) ────────────────────────
    CHUNK_SIZE: int = 512      # characters
    CHUNK_OVERLAP: int = 64


settings = Settings()
