"""Model layer (variant V6).

Every AI call goes through ``LLM`` and ``Embedder``, so the provider is a
configuration choice: ``stub`` for offline tests, ``litellm`` for the
company proxy, and later self-hosted open-source models or Claude.
"""

from secondbrain.config import Settings
from secondbrain.models.base import LLM, Embedder


def build_models(settings: Settings) -> tuple[LLM, Embedder]:
    if settings.model_provider == "stub":
        from secondbrain.models.stub import StubEmbedder, StubLLM

        return StubLLM(), StubEmbedder(settings.embedding_dim)
    if settings.model_provider == "litellm":
        from secondbrain.models.litellm import LiteLLMEmbedder, LiteLLMChat

        return LiteLLMChat(settings), LiteLLMEmbedder(settings)
    raise ValueError(f"Unknown model provider {settings.model_provider!r}")


__all__ = ["LLM", "Embedder", "build_models"]
