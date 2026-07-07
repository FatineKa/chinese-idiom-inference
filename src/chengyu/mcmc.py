"""mcmc.py — boucle Metropolis-Hastings. Python pur : elle ne manipule que
   des nombres renvoyés par scoring.py. Aucune dépendance à PyTorch/numpy ici."""

import math
import random
from chengyu.scoring import score_resume
from chengyu.prior   import log_prior


def log_poids(texte, idiome):
    return score_resume(texte, idiome) + log_prior(idiome)   # score_resume = PyTorch, en interne


def metropolis_hastings(texte, etat_init, proposer, n_pas, seed=0):
    rng    = random.Random(seed)
    actuel = etat_init
    lw     = log_poids(texte, actuel)
    trace  = []
    for _ in range(n_pas):
        candidat, log_hastings = proposer(actuel, rng)
        lw2 = log_poids(texte, candidat)                     # 1 appel Qwen (PyTorch dedans)
        if math.log(rng.random()) < (lw2 - lw) + log_hastings:
            actuel, lw = candidat, lw2
        trace.append(actuel)
    return trace


def proposeur_uniforme(idiomes):
    n = len(idiomes)
    def proposer(actuel, rng):
        while True:
            candidat = idiomes[rng.randrange(n)]
            if candidat != actuel:
                return candidat, 0.0
    return proposer