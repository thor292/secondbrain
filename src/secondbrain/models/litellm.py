"""Client for a LiteLLM proxy, which exposes an OpenAI-compatible API."""

import time

import httpx

from secondbrain.config import Settings

EMBED_BATCH = 100
RETRIES = 3


def _post(http: httpx.Client, path: str, body: dict) -> dict:
    """POST with retries: the proxy occasionally drops a connection or is briefly busy."""
    for attempt in range(RETRIES):
        try:
            response = http.post(path, json=body)
            if response.status_code in (429, 502, 503, 504) and attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.RemoteProtocolError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


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
        body = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        return _post(self._http, "/chat/completions", body)["choices"][0]["message"]["content"]


class LiteLLMEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.embedding_model:
            raise ValueError("SB_EMBEDDING_MODEL is not set")
        self.name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._http = _client(settings)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH):
            # Ask for our dimension explicitly: models with flexible output sizes
            # (e.g. gemini-embedding-001: 768 / 1536 / 3072) default to the largest.
            body = {"model": self.name, "input": texts[start:start + EMBED_BATCH], "dimensions": self.dim}
            data = sorted(_post(self._http, "/embeddings", body)["data"], key=lambda item: item["index"])
            vectors += [item["embedding"] for item in data]
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(
                    f"{self.name} returned {len(vector)} dimensions, "
                    f"but SB_EMBEDDING_DIM is {self.dim}"
                )
        return vectors
