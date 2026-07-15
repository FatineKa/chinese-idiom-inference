"""prior.py — le prior log p(idiome), estimé par les fréquences d'usage."""

import math
import json

# Fichier {idiome: nombre d'occurrences}. Construit par scripts/00_build_freq.py
with open("data/freq_idiomes.json", encoding="utf-8") as f:
    _freq = json.load(f)

_total = sum(_freq.values())
_N     = len(_freq)


def log_prior(idiome: str) -> float:
    """log p(i) avec lissage de Laplace : (compte + 1) / (total + N).
       Le +1 évite log(0) pour un idiome jamais vu et donne une petite
       masse aux idiomes rares (sans les exclure)."""
    c = _freq.get(idiome, 0)
    return math.log((c + 1) / (_total + _N))