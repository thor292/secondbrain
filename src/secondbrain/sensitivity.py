from enum import IntEnum


class Sensitivity(IntEnum):
    """Ordered labels: a grant up to a level allows every level below it."""

    PUBLIC = 0
    INTERNAL = 1
    RESTRICTED = 2

    @classmethod
    def parse(cls, name: str) -> "Sensitivity":
        return cls[name.upper()]
