"""04_argmax_eval.py — évaluation en dictionnaire complet (plus de QCM).
Pour chaque texte : classement exact de TOUS les idiomes, top-1 / top-10,
avec et sans prior. On rapporte aussi le rang de la cible."""
import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.argmax import scores_texte
from chengyu.prior import log_prior

N = 50    # dictionnaire complet => coûteux : commencer petit
dico, longueurs = charger_dico()
liste_idiomes = list(dico)
df = pd.read_csv("data/raw/cip/train.csv")

t1s = t1a = t10s = t10a = evalues = sautes = 0
for src, dst in zip(df["src"], df["dst"]):
    if evalues >= N:
        break
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible is None:
        sautes += 1
        continue
    texte = normaliser(dst)
    vrais = scores_texte(texte, liste_idiomes)
    avec  = {i: v + log_prior(i) for i, v in vrais.items()}
    rs = sorted(vrais, key=vrais.get, reverse=True).index(cible) + 1
    ra = sorted(avec,  key=avec.get,  reverse=True).index(cible) + 1
    t1s += rs == 1; t10s += rs <= 10
    t1a += ra == 1; t10a += ra <= 10
    evalues += 1
    print(f"[{evalues}/{N}] cible={cible}  rang sans/avec prior : {rs}/{ra}")

print(f"\névalués : {evalues} (sautés : {sautes})")
print(f"top-1  sans prior : {t1s/evalues:.1%}   avec prior : {t1a/evalues:.1%}")
print(f"top-10 sans prior : {t10s/evalues:.1%}  avec prior : {t10a/evalues:.1%}")