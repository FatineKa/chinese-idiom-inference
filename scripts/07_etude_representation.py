"""07_etude_representation.py — Δ_l(i,t) sépare-t-il l'idiome correct des
distracteurs, et à quelle couche l le mieux ? Point 1 de la liste
"to be done" (statistical study on the embedding modifications).

Sur un échantillon de train.csv, pas sur les 31 113 idiomes : chaque paire
coûte 2 forward pass (idiome correct + leurre), donc rester petit ici."""
import random
import pandas as pd
from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.representation import modification_par_couche

N = 100

print(f"chargement du dictionnaire et de train.csv...")
dico, longueurs = charger_dico()
liste_idiomes = list(dico)
df = pd.read_csv("data/raw/cip/train.csv")
rng = random.Random(0)
print(f"dictionnaire : {len(liste_idiomes)} idiomes — objectif : {N} paires évaluées\n")

n_couches = None
gagnes = None       # gagnes[l] = nb de paires où le correct a Δ_l plus faible que le leurre
evalues = sautes = 0

for src, dst in zip(df["src"], df["dst"]):
    if evalues >= N:
        break
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible is None:
        sautes += 1
        continue

    texte = normaliser(dst)
    leurre = rng.choice(liste_idiomes)

    delta_correct = modification_par_couche(cible, texte)
    delta_leurre = modification_par_couche(leurre, texte)

    if gagnes is None:
        n_couches = len(delta_correct)
        gagnes = [0] * n_couches
        print(f"{n_couches} couches détectées (0 = embeddings, {n_couches - 1} = dernière couche)\n")

    victoires_ici = 0
    for l in range(n_couches):
        if delta_correct[l] < delta_leurre[l]:
            gagnes[l] += 1
            victoires_ici += 1
    evalues += 1

    print(f"[{evalues}/{N}] cible={cible}  leurre={leurre}  "
          f"couches où le correct gagne : {victoires_ici}/{n_couches}")

print(f"\névalués : {evalues}  (sautés : {sautes})\n")
print("taux de séparation par couche (proportion de paires où Δ_correct < Δ_leurre) :")
for l, g in enumerate(gagnes):
    barre = "█" * round(40 * g / evalues)
    print(f"  couche {l:2d} : {g/evalues:5.1%}  {barre}")

meilleure = max(range(n_couches), key=lambda l: gagnes[l])
print(f"\nmeilleure couche : {meilleure}  ({gagnes[meilleure]/evalues:.1%} de séparation)")
if gagnes[meilleure] / evalues < 0.6:
    print("attention : même la meilleure couche sépare à peine mieux que le hasard (50%) —")
    print("le signal est peut-être trop faible pour informer un proposeur MCMC en l'état.")
