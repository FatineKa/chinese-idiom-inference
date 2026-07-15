import math
import json
from collections import Counter

import pandas as pd

from chengyu.mcmc import metropolis_hastings, proposeur_uniforme, log_poids
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome


def verifier(texte, sous_ensemble, n_pas=20000):
    """Compare la posterieure exacte (calculable sur un petit ensemble)
    aux frequences de visite du MCMC. Si ca coincide, le MCMC est correct."""
    w = {i: math.exp(log_poids(texte, i)) for i in sous_ensemble}
    Z = sum(w.values())
    exacte = {i: w[i] / Z for i in sous_ensemble}      # posterieure EXACTE

    prop = proposeur_uniforme(sous_ensemble)
    trace = metropolis_hastings(texte, sous_ensemble[0], prop, n_pas)
    trace = trace[n_pas // 10:]                        # burn-in
    c = Counter(trace)
    print(f"{'idiome':<8} {'exacte':>8} {'MCMC':>8}")
    for i in sorted(sous_ensemble, key=lambda x: -exacte[x]):
        print(f"{i:<8} {exacte[i]:>8.3f} {c[i] / len(trace):>8.3f}")


if __name__ == "__main__":
    # 1. un texte propre et sa cible
    df = pd.read_csv("data/raw/cip/train.csv")
    dico, longueurs = charger_dico()
    for src, dst in zip(df["src"], df["dst"]):
        cible = trouver_idiome(src, dst, dico, longueurs)
        if cible:
            break
    texte = normaliser(dst)
    print("texte :", texte)
    print("cible :", cible, "\n")

    # 2. 20 candidats : la cible + 19 idiomes frequents
    with open("data/freq_idiomes.json", encoding="utf-8") as f:
        freq = json.load(f)
    frequents = sorted(freq, key=freq.get, reverse=True)
    sous_ensemble = [cible] + [i for i in frequents if i != cible][:19]

    # 3. exacte vs MCMC
    verifier(texte, sous_ensemble, n_pas=20000)