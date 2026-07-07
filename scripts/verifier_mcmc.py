"""verifier_mcmc.py — prouve que le MCMC échantillonne la bonne loi,
   en comparant, sur un petit sous-ensemble, la posterior exacte aux
   fréquences visitées."""

import math
from collections import Counter
from chengyu.mcmc import metropolis_hastings, log_poids


def verifier(texte, sous_ensemble, n_pas=20000):
    # 1) posterior EXACTE : possible car le sous-ensemble est petit (Z calculable)
    w = {i: math.exp(log_poids(texte, i)) for i in sous_ensemble}
    Z = sum(w.values())
    exacte = {i: w[i] / Z for i in sous_ensemble}

    # 2) MCMC sur le même sous-ensemble
    trace = metropolis_hastings(texte, sous_ensemble, n_pas)
    trace = trace[n_pas // 10:]                        # jeter le burn-in (10 %)

    # 3) fréquences visitées
    c = Counter(trace)
    freq = {i: c[i] / len(trace) for i in sous_ensemble}

    # 4) comparer : les deux colonnes doivent coïncider -> MCMC correct
    for i in sous_ensemble:
        print(i, "exacte=%.3f" % exacte[i], "MCMC=%.3f" % freq.get(i, 0.0))