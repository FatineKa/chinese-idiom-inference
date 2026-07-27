"""05_rademacher_bound.py — calcule les constantes réelles du chapitre de
sélection de modèle (b, lambda(K), R_hat(K)) à partir des données du projet,
et mesure le coût effectif de l'élagage du dictionnaire par prior.

Ne modifie ni argmax.py ni prior.py : script autonome, lecture seule.
Répond à deux problèmes ouverts du chapitre : aucune constante numérique
calculée, et aucune validation empirique."""
import math
import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.argmax import scores_texte
from chengyu.prior import log_prior

N_EVAL = 20            # nombre de textes évalués (chaque texte coûte un score sur tout le dictionnaire)
DELTA = 0.05           # confiance 1 - delta
K_CANDIDATS_BASE = [50, 100, 250, 500, 1000, 2000, 5000, 10000]

dico, longueurs = charger_dico()
liste_idiomes = list(dico)
N = len(liste_idiomes)
K_candidats = sorted(set(K for K in K_CANDIDATS_BASE if K < N) | {N})
M = len(K_candidats)

# I_K : dictionnaires emboîtés, triés par prior décroissant (une seule fois, section 5 du chapitre)
idiomes_tries = sorted(liste_idiomes, key=log_prior, reverse=True)
idiomes_par_K = {K: idiomes_tries[:K] for K in K_candidats}
ensembles_par_K = {K: set(idiomes_par_K[K]) for K in K_candidats}

df = pd.read_csv("data/raw/cip/train.csv")

h_par_K = {K: [] for K in K_candidats}    # h_K(t_j) pour chaque texte évalué
couverts = {K: 0 for K in K_candidats}    # la cible est-elle dans I_K ?
g_min, g_max = math.inf, -math.inf

evalues = sautes = 0
for src, dst in zip(df["src"], df["dst"]):
    if evalues >= N_EVAL:
        break
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible is None:
        sautes += 1
        continue
    texte = normaliser(dst)

    vrais = scores_texte(texte, liste_idiomes)                # log p(t|i), tous les idiomes
    g = {i: -v - log_prior(i) for i, v in vrais.items()}      # g_i(t) = ell_i(t), section 3

    for val in g.values():
        g_min = min(g_min, val)
        g_max = max(g_max, val)

    for K in K_candidats:
        h_par_K[K].append(min(g[i] for i in idiomes_par_K[K]))
        if cible in ensembles_par_K[K]:
            couverts[K] += 1

    evalues += 1
    print(f"[{evalues}/{N_EVAL}] cible={cible}  g_min/max observés jusqu'ici : {g_min:.2f} / {g_max:.2f}")

print(f"\névalués : {evalues} (sautés : {sautes})")
if evalues == 0:
    raise SystemExit("Aucun texte évalué (tout a été sauté) — vérifier data/raw/cip/train.csv.")

m = evalues
b = g_max   # borne empirique -- pas une borne pire cas prouvée (cf. Hypothèse 1.5 du chapitre)
print(f"\nb (empirique, = max observé de g_i(t)) : {b:.3f}")
print(f"min observé (doit être proche de 0, Lemme 1.6) : {g_min:.3f}")

print(f"\n{'K':>7} {'couverture':>11} {'R_hat(K)':>10} {'lambda(K)':>11} {'borne R(K)':>11} {'lambda_Tal':>11}")
for K in K_candidats:
    R_hat = sum(h_par_K[K]) / m
    lam = 2 * b * math.sqrt(2 * math.log(M) / m)
    borne = R_hat + 2 * lam

    variance_empirique = sum((h - R_hat) ** 2 for h in h_par_K[K]) / m
    lam_tal = (2 * b * math.sqrt(2 * math.log(M) / m)
               + math.sqrt(2 * variance_empirique * math.log(4 / DELTA) / m)
               + 2 * b * math.log(4 / DELTA) / (3 * m))

    print(f"{K:>7} {couverts[K] / m:>10.1%} {R_hat:>10.3f} {lam:>11.3f} {borne:>11.3f} {lam_tal:>11.3f}")

print(f"""
Lecture :
- « couverture » : fraction des textes où l'idiome cible fait partie du
  dictionnaire élagué I_K. C'est une condition nécessaire, pas la borne du
  chapitre elle-même -- si la couverture est basse pour un K donné, élaguer
  à ce K perd la bonne réponse d'entrée de jeu, quel que soit lambda(K).
- R_hat(K) + 2*lambda(K) borne R(K) avec probabilité >= 1 - delta (Théorème 1.14).
- lambda_Tal utilise la variance EMPIRIQUE de h_K, pas la borne pire cas de
  Popoviciu (b^2/4) : cf. Corollaire 1.19 / Remarque 1.20 du chapitre. Sur un
  échantillon de {N_EVAL} textes, cette variance empirique est elle-même
  bruitée -- ne pas sur-interpréter sans un échantillon plus grand.
- b est une borne EMPIRIQUE (max observé), pas la borne pire cas de
  l'Hypothèse 1.5 (qui dépend de epsilon, non mesuré ici). Un texte ou un
  idiome hors échantillon pourrait donner un g_i(t) plus grand.
""")
