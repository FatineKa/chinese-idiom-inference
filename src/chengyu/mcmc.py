"""mcmc.py — boucle Metropolis-Hastings. proposeur_uniforme est du Python
   pur (aucune dépendance à PyTorch/numpy). proposeur_par_texte, plus bas,
   dépend de numpy et, via representation.py, de PyTorch — mais un seul
   forward pass Qwen à la construction, indépendant du nombre d'idiomes."""

import math
import random
import numpy as np
from chengyu.scoring import score_resume
from chengyu.prior   import log_prior
from chengyu.representation import etat_texte_par_couche


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


def proposeur_par_texte(idiomes: list, texte: str, embeddings_statiques: np.ndarray,
                         couche: int, temperature: float = 1.0):
    """Proposeur informé, coût constant : un seul forward pass Qwen (sur le
    texte seul, ici à la construction), comparé aux embeddings statiques des
    idiomes précalculés une fois (geometry.plongements). `couche` doit venir
    de 07_etude_representation.py — ne pas en choisir une au hasard."""
    v = etat_texte_par_couche(texte)[couche].numpy()
    normes = np.linalg.norm(embeddings_statiques, axis=1) * np.linalg.norm(v)
    poids = np.exp((embeddings_statiques @ v / normes) / temperature)
    poids /= poids.sum()
    index = {idiome: k for k, idiome in enumerate(idiomes)}

    print(f"proposeur informé construit : couche {couche}, {len(idiomes)} idiomes, "
          f"poids min={poids.min():.2e} max={poids.max():.2e} "
          f"(idiome le plus favorisé : {idiomes[poids.argmax()]})")

    def proposer(actuel, rng):
        idx = rng.choices(range(len(idiomes)), weights=poids.tolist())[0]
        candidat = idiomes[idx]
        log_hastings = math.log(poids[index[actuel]]) - math.log(poids[idx])
        return candidat, log_hastings
    return proposer