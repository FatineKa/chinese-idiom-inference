"""06_gmm_embeddings_bound.py — validation empirique du chapitre 12 (cas continu
adaptatif) : sélection du nombre de composantes K d'un mélange gaussien ajusté par
EM sur des plongements d'idiomes, avec la borne par recouvrement de Théorème 1.11.

Lecture seule sur le reste du dépôt ; écrit uniquement dans les variables locales
et la sortie standard."""
import math
import random  # noqa: F401 (SEED reproducibility)

import numpy as np
from sklearn.mixture import GaussianMixture

from chengyu.evaluation import charger_dico
from chengyu.geometry import plongements, reduire
from chengyu.gmm_select import lambda_K

M_ECH = 600          # nombre d'idiomes échantillonnés (m)
P = 6                # dimension après PCA
K_CANDIDATS = [1, 2, 3, 4, 5, 6, 8, 10, 14]
REG_COVAR = 0.3      # plancher pratique sur les valeurs propres de Sigma (proxy de sigma_min^2)
R_PERCENTILE = 90    # rayon = ce centile de la norme empirique, pas le max (Hypothèse : les
                     # idiomes hors de ce rayon sont hors du support compact modélisé)
SEED = 0

random.seed(SEED)
np.random.seed(SEED)

dico, _ = charger_dico()
idiomes = random.sample(sorted(dico), M_ECH)
print(f"Plongements de {M_ECH} idiomes (table d'entrée Qwen, moyenne sur les tokens)...")
X = plongements(idiomes)
print("X brut :", X.shape)

X_red, pca = reduire(X, P)
print(f"PCA -> {P} dims, variance expliquée : {pca.explained_variance_ratio_.sum():.1%}")

# Standardiser pour que R, sigma_min^2, sigma_max^2 soient sur une échelle raisonnable
X_red = (X_red - X_red.mean(axis=0)) / X_red.std(axis=0)

normes = np.linalg.norm(X_red, axis=1)
R_empirique = float(np.percentile(normes, R_PERCENTILE))
print(f"R (centile {R_PERCENTILE} de la norme empirique -- Hypothèse de support compact) : "
      f"{R_empirique:.3f}  (max observé : {normes.max():.3f})")

resultats = []
for K in K_CANDIDATS:
    if K >= M_ECH:
        continue
    gmm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=REG_COVAR,
                           random_state=SEED, max_iter=200)
    gmm.fit(X_red)
    log_vraisemblance = gmm.score(X_red)          # moyenne de log p(x), déjà /m
    R_hat = -log_vraisemblance                     # R_hat(K) = perte moyenne = Def. 1.11 (continu)

    eigs = np.linalg.eigvalsh(gmm.covariances_)
    sigma_min2 = max(float(eigs.min()), REG_COVAR)
    sigma_max2 = float(eigs.max())

    info = lambda_K(K, P, R_empirique, sigma_min2, sigma_max2, M_ECH)
    lam_affichable = info["lambda"] if math.isfinite(info["lambda"]) else float("inf")
    borne = R_hat + 2 * lam_affichable if math.isfinite(lam_affichable) else float("inf")

    resultats.append((K, info["d_K"], R_hat, lam_affichable, borne, info["log_lambda"]))
    lam_txt = f"{lam_affichable:10.3e}" if math.isfinite(lam_affichable) else f"exp({info['log_lambda']:.1f})"
    print(f"[K={K:>2}] d_K={info['d_K']:>4}  R_hat={R_hat:8.3f}  lambda={lam_txt:>16}  "
          f"sigma^2 in [{sigma_min2:.4f}, {sigma_max2:.4f}]")

print(f"\n{'K':>3} {'d_K':>6} {'R_hat(K)':>10} {'log(lambda(K))':>16}")
for K, d_K, R_hat, lam, borne, log_lam in resultats:
    print(f"{K:>3} {d_K:>6} {R_hat:>10.3f} {log_lam:>16.2f}")

# Le terme lambda(K) domine ici R_hat(K) de plusieurs ordres de grandeur (voir section 12.5
# du chapitre) : la borne est vraie mais non informative en valeur absolue. Ce qui reste
# vérifiable et intéressant, c'est la FORME de la dépendance en K -- via d_K -- comparée à
# celle, plate, du chapitre discret (Remarque 1.12).
K_etoile_rademacher = min(resultats, key=lambda r: r[5])[0]  # argmin log(lambda(K)) seul
print(f"\nargmin_K log(lambda(K)) seul = {K_etoile_rademacher} "
      f"(indicatif : montre que le terme de pénalité varie bien avec K, via d_K).")
