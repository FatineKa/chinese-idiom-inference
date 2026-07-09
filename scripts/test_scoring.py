import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.scoring import score_resume

df = pd.read_csv("data/raw/cip/train.csv")
dico, longueurs = charger_dico()

# première ligne "propre" du corpus
for src, dst in zip(df["src"], df["dst"]):
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible:
        break
texte = normaliser(dst)

print("texte  :", texte)
print("cible  :", cible)
print("score(cible)     :", score_resume(texte, cible))
print("score(hors sujet):", score_resume(texte, "无论如何"))