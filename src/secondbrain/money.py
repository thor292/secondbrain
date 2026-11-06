"""Rupees the Indian way, for every text the brain writes: ₹45,600, ₹3.4 lakh, ₹2.15 crore."""


def group(n: int) -> str:
    """Indian digit grouping: 12345678 -> 1,23,45,678."""
    s = str(abs(n))
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ("-" if n < 0 else "") + ",".join(parts + [tail])


def inr(x: float) -> str:
    """Exact amount: ₹1,23,456."""
    return ("-" if x < 0 else "") + "₹" + group(round(abs(x)))


def inr_short(x: float) -> str:
    """Rounded for headlines: ₹45,600, ₹3.4 lakh, ₹2.15 crore."""
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:.2f} crore"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:.1f} lakh"
    return sign + "₹" + group(round(x))
