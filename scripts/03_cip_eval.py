"""03_cip_eval.py — évaluation QCM à 7 candidats.
   Entrée = dst normalisé (sans l'idiome -> pas de fuite).
   Cible  = idiome extrait par trouver_idiome.
   Compare la précision avec prior vs sans prior (vraisemblance seule)."""

import random
import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.scoring import score_resume
from chengyu.prior   import log_prior

N = 200                 # nombre d'exemples évalués
random.seed(0)

dico, longueurs = charger_dico()
liste_idiomes = list(dico)
df = pd.read_csv("data/raw/cip/train.csv")

bon_sans, bon_avec, evalues, sautes = 0, 0, 0, 0

for src, dst in zip(df["src"], df["dst"]):
    if evalues >= N:
        break
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible is None:                      # ligne ambiguë -> on saute
        sautes += 1
        continue

    texte = normaliser(dst)                # entrée = paraphrase sans l'idiome
    # 7 candidats : le bon + 6 tirés au hasard (distincts du bon)
    distracteurs = random.sample([i for i in liste_idiomes if i != cible], 6)
    candidats = distracteurs + [cible]
    random.shuffle(candidats)

    # score de chaque candidat
    vrais  = {i: score_resume(texte, i) for i in candidats}   # log p(texte | i)
    # sans prior : argmax de la vraisemblance seule
    pred_sans = max(candidats, key=lambda i: vrais[i])
    # avec prior : argmax de log w = vraisemblance + prior
    pred_avec = max(candidats, key=lambda i: vrais[i] + log_prior(i))

    bon_sans += (pred_sans == cible)
    bon_avec += (pred_avec == cible)
    evalues  += 1

print(f"évalués : {evalues}  (lignes sautées : {sautes})")
print(f"précision sans prior : {bon_sans}/{evalues} = {bon_sans/evalues:.1%}")
print(f"précision avec prior : {bon_avec}/{evalues} = {bon_avec/evalues:.1%}")