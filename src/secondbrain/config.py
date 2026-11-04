from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SB_", env_file=".env", extra="ignore")

    admin_database_url: str = (
        "postgresql://secondbrain_admin:dev_admin_password@localhost:55432/secondbrain"
    )
    app_database_url: str = (
        "postgresql://secondbrain_app:dev_app_password@localhost:55432/secondbrain"
    )
    redis_url: str = "redis://localhost:56379/0"
    # Relationship layer. Each company gets its own graph, named
    # "<graph_prefix>_<company id>", so tests use a different prefix.
    falkordb_url: str = "redis://localhost:56380"
    graph_prefix: str = "sb"

    model_provider: str = "stub"  # "stub" or "litellm"
    litellm_base_url: str = ""
    litellm_api_key: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1024

    short_term_ttl: int = 86400
    # Cosine distance above which a department result counts as "not found",
    # triggering fallback to the company brain.
    fallback_max_distance: float = 0.6
    search_limit: int = 5

    # Admin API (onboarding, document upload) is disabled while this is empty.
    admin_token: str = ""
    # Demo mode lists users so the chat screen can switch between them. In
    # production, identity comes from SSO instead of an X-User-Id header.
    demo_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
