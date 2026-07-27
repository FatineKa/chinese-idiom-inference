"""08_chainage_birge_massart.py — compare, sur les memes plongements et ajustements
GMM que le script 06, quatre variantes de penalite pour la selection de K :

  1. lambda(K)        Theoreme 1.12, eps=1/m arbitraire (deja dans gmm_select.py)
  2. lambda(K), eps*   Corollaire 1.22 : meme theoreme, eps choisi pour equilibrer
                       les deux termes (eps* = b/(4 L_K)) au lieu de 1/m
  3. lambda_chainage(K) Theoreme 1.22 : chainage a echelles multiples sous eps*
  4. pen_BM(K)          Birge-Massart (section 13) : mecanisme different (Hellinger),
                       borne un risque different (non directement comparable a 1-3)

Repond a la question ouverte de la section 11.4 du chapitre (et a la Remarque
1.21) : le chainage resout-il la mollesse de la borne, et de combien ? Lecture
seule sur le reste du depot."""
import random

import numpy as np
from sklearn.mixture import GaussianMixture

from chengyu.evaluation import charger_dico
from chengyu.geometry import plongements, reduire
from chengyu.gmm_select import (bornes_g, dimension_parametres, epsilon_optimal,
                                 lambda_K, lambda_K_chainage, penalite_birge_massart)

M_ECH = 600
P = 6
K_CANDIDATS = [1, 2, 3, 4, 5, 6, 8, 10, 14]
REG_COVAR = 0.3
R_PERCENTILE = 90
SEED = 0

random.seed(SEED)
np.random.seed(SEED)

dico, _ = charger_dico()
idiomes = random.sample(sorted(dico), M_ECH)
print(f"Plongements de {M_ECH} idiomes (memes reglages que le script 06)...")
X = plongements(idiomes)
X_red, pca = reduire(X, P)
X_red = (X_red - X_red.mean(axis=0)) / X_red.std(axis=0)

normes = np.linalg.norm(X_red, axis=1)
R_empirique = float(np.percentile(normes, R_PERCENTILE))
print(f"R (centile {R_PERCENTILE}) = {R_empirique:.3f}\n")

lignes = []
for K in K_CANDIDATS:
    if K >= M_ECH:
        continue
    gmm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=REG_COVAR,
                           random_state=SEED, max_iter=200)
    gmm.fit(X_red)
    R_hat = -gmm.score(X_red)

    eigs = np.linalg.eigvalsh(gmm.covariances_)
    sigma_min2 = max(float(eigs.min()), REG_COVAR)
    sigma_max2 = float(eigs.max())

    info_orig = lambda_K(K, P, R_empirique, sigma_min2, sigma_max2, M_ECH)
    eps_etoile = epsilon_optimal(K, P, R_empirique, sigma_min2, sigma_max2)
    info_opt = lambda_K(K, P, R_empirique, sigma_min2, sigma_max2, M_ECH, eps=eps_etoile)
    info_chain = lambda_K_chainage(K, P, R_empirique, sigma_min2, sigma_max2, M_ECH)
    info_bm = penalite_birge_massart(K, P, M_ECH)

    lignes.append(dict(K=K, d_K=dimension_parametres(K, P), R_hat=R_hat,
                        lam_orig=info_orig["lambda"], lam_opt=info_opt["lambda"],
                        lam_chain=info_chain["lambda"], pen_bm=info_bm["penalite"],
                        eps_etoile=eps_etoile))
    print(f"[K={K:>2}] d_K={dimension_parametres(K, P):>4}  R_hat={R_hat:8.3f}  "
          f"sigma^2 in [{sigma_min2:.4f}, {sigma_max2:.4f}]  eps*={eps_etoile:.3e}")

print(f"\n{'K':>3} {'d_K':>5} {'R_hat':>8} {'lam(eps=1/m)':>14} {'lam(eps*)':>12} "
      f"{'lam_chainage':>13} {'pen_BM':>9}")
for l in lignes:
    def fmt(x):
        return f"{x:.3e}" if (x != float('inf') and (x >= 1e4 or 0 < x < 1e-3)) else f"{x:.3f}"
    print(f"{l['K']:>3} {l['d_K']:>5} {l['R_hat']:>8.3f} {fmt(l['lam_orig']):>14} "
          f"{fmt(l['lam_opt']):>12} {fmt(l['lam_chain']):>13} {fmt(l['pen_bm']):>9}")

print("""
Lecture :
- lam(eps=1/m)  : Theoreme 1.12 tel qu'utilise jusqu'ici (script 06) -- eps=1/m
  est une commodite de calcul, pas un choix optimise.
- lam(eps*)     : meme theoreme, eps choisi pour equilibrer ses deux termes
  (Corollaire 1.22 du chapitre, section 12.1). Aucun nouvel outil : juste un
  meilleur choix du parametre libre deja present dans le Theoreme 1.12.
- lam_chainage  : chainage a echelles multiples sous eps* (Theoreme 1.22,
  section 12.2) -- le raffinement proprement dit, sur la base d'eps* et non du
  diametre de Theta_K (qui redemanderait la borne triviale a chaque niveau
  grossier avant d'atteindre eps*, un piege identifie en ecrivant ce chapitre).
- pen_BM        : Birge-Massart (section 13) -- mecanisme different (entropie de
  Hellinger, pas de facteur D_max/D_min), mais borne un risque de Hellinger, pas
  R(K) : pas directement comparable en valeur aux trois colonnes precedentes.
""")

argmin_orig = min(lignes, key=lambda l: l["R_hat"] + 2 * l["lam_orig"])["K"]
argmin_opt = min(lignes, key=lambda l: l["R_hat"] + 2 * l["lam_opt"])["K"]
argmin_chain = min(lignes, key=lambda l: l["R_hat"] + 2 * l["lam_chain"])["K"]
print(f"argmin_K [R_hat(K) + 2*lambda(K)] : eps=1/m -> K={argmin_orig}  "
      f"eps* -> K={argmin_opt}  chainage -> K={argmin_chain}")
if len({argmin_orig, argmin_opt, argmin_chain}) == 1:
    print("Les trois variantes recommandent le meme K : le raffinement ne change\n"
          "pas la decision sur cet echantillon, meme s'il resserre la borne.")
else:
    print("Les variantes ne s'accordent PAS sur K : le choix du raffinement change\n"
          "la decision pratique sur cet echantillon -- a documenter honnetement.")
