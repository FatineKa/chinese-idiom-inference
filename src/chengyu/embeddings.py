"""embeddings.py — proposition guidée. Python pur. Les tableaux voisins/Kmat/Zrow
   sont des listes ordinaires, précalculées une fois (voir build_embeddings ci-dessous)."""

import math


def proposeur_embeddings(idiomes, voisins, Kmat, Zrow):
    index = {ch: k for k, ch in enumerate(idiomes)}          # idiome -> indice

    def log_q(i, j):
        pos = voisins[i].index(j)                            # position de j dans les voisins de i
        return math.log(Kmat[i][pos] / Zrow[i])

    def proposer(actuel, rng):
        i = index[actuel]
        j = rng.choices(voisins[i], weights=Kmat[i])[0]      # tire un voisin selon K(i, .)
        candidat = idiomes[j]
        log_hastings = log_q(j, i) - log_q(i, j)
        return candidat, log_hastings

    return proposer