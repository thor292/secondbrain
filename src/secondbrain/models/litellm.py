"""Client for a LiteLLM proxy, which exposes an OpenAI-compatible API."""

import httpx

from secondbrain.config import Settings


def _client(settings: Settings) -> httpx.Client:
    if not settings.litellm_base_url:
        raise ValueError("SB_LITELLM_BASE_URL is not set")
    return httpx.Client(
        base_url=settings.litellm_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.litellm_api_key}"},
        timeout=60.0,
    )


class LiteLLMChat:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_model:
            raise ValueError("SB_LLM_MODEL is not set")
        self.name = settings.llm_model
        self._http = _client(settings)

    def complete(self, system: str, prompt: str) -> str:
        response = self._http.post(
            "/chat/completions",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LiteLLMEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.embedding_model:
            raise ValueError("SB_EMBEDDING_MODEL is not set")
        self.name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._http = _client(settings)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._http.post("/embeddings", json={"model": self.name, "input": texts})
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(
                    f"{self.name} returned {len(vector)} dimensions, "
                    f"but SB_EMBEDDING_DIM is {self.dim}"
                )
        return vectors
