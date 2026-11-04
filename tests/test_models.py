import math

from secondbrain import documents, tree
from secondbrain.models.stub import StubEmbedder, StubLLM
from secondbrain.schema import vector_type
from secondbrain.service import CLASSIFY_SYSTEM


def cosine_distance(a, b):
    return 1 - sum(x * y for x, y in zip(a, b))


def test_stub_embedder_matches_shared_words():
    embedder = StubEmbedder(256)
    question, related, unrelated = embedder.embed(
        ["What is the travel reimbursement limit?",
         "The travel reimbursement limit is 500 dollars per trip.",
         "Interviews have at least two rounds."]
    )
    assert math.isclose(sum(v * v for v in question), 1.0)
    assert cosine_distance(question, related) < cosine_distance(question, unrelated)


def test_stub_llm_classifies_memory_types():
    llm = StubLLM()
    assert '"user"' in llm.complete(CLASSIFY_SYSTEM, "Ravi prefers email")
    assert '"episodic"' in llm.complete(CLASSIFY_SYSTEM, "Met the vendor yesterday")
    assert '"semantic"' in llm.complete(CLASSIFY_SYSTEM, "Refund policy is 30 days")


def test_company_of_never_returns_root():
    assert tree.company_of("root.acme.sales") == "root.acme"
    assert tree.company_of("root.acme") == "root.acme"


def test_vector_type_respects_index_limits():
    assert vector_type(1024) == ("vector(1024)", "vector_cosine_ops")
    assert vector_type(3072) == ("halfvec(3072)", "halfvec_cosine_ops")


def test_chunk_splits_on_paragraphs():
    text = "\n\n".join(["a" * 500] * 5)
    chunks = documents.chunk(text, max_chars=1200)
    assert len(chunks) == 3
    assert all(len(c) <= 1200 for c in chunks)
