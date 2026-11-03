from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
