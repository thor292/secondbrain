from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SB_", env_file=".env", extra="ignore")

    admin_database_url: str = (
        "postgresql://secondbrain_admin:dev_admin_password@localhost:55432/secondbrain"
    )
    # The API's own login, which row-level security applies to. Leave it empty to
    # use one login for everything: the API then connects with admin_database_url
    # and switches every connection to app_role, so row-level security still applies.
    # That login must be a member of app_role (GRANT secondbrain_app TO <login>).
    app_database_url: str = ""
    app_role: str = "secondbrain_app"
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

    # Live notifications to people (a second MQTT broker, separate from data
    # ingest): empty host switches them off. Per-user broker passwords are
    # derived from live_secret; see docs/LIVE.md.
    live_host: str = ""
    live_port: int = 51884
    live_ws_url: str = "ws://localhost:59001"
    live_secret: str = ""
    live_publisher_password: str = ""

    def check_required(self) -> None:
        """Stop startup with one clear message instead of a stack of connection errors."""
        problems = []
        if not self.admin_database_url:
            problems.append("SB_ADMIN_DATABASE_URL is not set")
        if self.model_provider == "litellm":
            for name in ("litellm_base_url", "llm_model", "embedding_model"):
                if not getattr(self, name):
                    problems.append(f"SB_{name.upper()} is not set (SB_MODEL_PROVIDER=litellm)")
        if problems:
            raise RuntimeError("SecondBrain can't start: " + "; ".join(problems))

    def app_connection(self) -> tuple[str, str | None]:
        """Where the API connects, and the role to switch to (None: the login is already it)."""
        if self.app_database_url:
            return self.app_database_url, None
        return self.admin_database_url, self.app_role


@lru_cache
def get_settings() -> Settings:
    return Settings()
