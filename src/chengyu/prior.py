"""prior.py — the prior log p(idiom), estimated from usage frequencies."""

import json
import math

# File {idiom: occurrence count}. Built by scripts/00_build_freq.py
with open("data/idiom_freq.json", encoding="utf-8") as f:
    _freq = json.load(f)

_total = sum(_freq.values())
_N     = len(_freq)


def log_prior(idiom: str) -> float:
    """log p(i) with Laplace smoothing: (count + 1) / (total + N).
       The +1 avoids log(0) for an idiom never seen, and gives a small
       amount of mass to rare idioms (without excluding them)."""
    c = _freq.get(idiom, 0)
    return math.log((c + 1) / (_total + _N))
