"""build_freq.py — compte la fréquence des idiomes dans CIP (train.csv) et
   écrit data/freq_idiomes.json. À lancer depuis la racine ~/projet-memoire."""

import json
from collections import Counter
import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, idiomes_presents

dico, longueurs = charger_dico()
df = pd.read_csv("data/raw/cip/train.csv")

# Prior = fréquence d'usage : on compte tous les idiomes vus dans les src.
# (plus de données qu'en ne comptant que l'idiome cible, prior plus robuste)
compte = Counter()
for src in df["src"]:
    for idiome in idiomes_presents(normaliser(src), dico, longueurs):
        compte[idiome] += 1

with open("data/freq_idiomes.json", "w", encoding="utf-8") as f:
    json.dump(dict(compte), f, ensure_ascii=False)

print("idiomes comptés :", len(compte), "| top :", compte.most_common(5))