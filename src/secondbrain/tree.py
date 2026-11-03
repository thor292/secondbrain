"""Brain tree paths.

Brains are addressed by ltree paths: ``root``, ``root.company_a``,
``root.company_a.sales``. Level 1 is our organization's root brain, level 2 a
company, level 3+ departments and below.
"""

ROOT = "root"
COMPANY_LEVEL = 2


def level(path: str) -> int:
    return len(path.split("."))


def company_of(path: str) -> str:
    """The company brain a path belongs to. Queries never go above this."""
    if level(path) < COMPANY_LEVEL:
        raise ValueError(f"{path!r} is not inside a company")
    return ".".join(path.split(".")[:COMPANY_LEVEL])
