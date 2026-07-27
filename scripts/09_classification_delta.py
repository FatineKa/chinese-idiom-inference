"""09_classification_delta.py — etude exploratoire PAR CLASSIFICATION (accord du
directeur, 2026-07-27), qui remplace/complete le script 07 pour repondre a une
question precise : le signal

    Delta_l(i, t) = 1 - cos( e_statique(i), e_contextualisee(i | t, couche l) )

(representation.py, section 11 du chapitre) permet-il de DISTINGUER l'idiome
correct des autres idiomes candidats, pour un texte donne t ?

Le script 07 posait deja cette question, mais avec une seule comparaison par
texte (le correct contre UN leurre) et une seule statistique descriptive (le
"taux de victoire" : le correct a-t-il un Delta_l plus petit que le leurre,
oui/non). C'est un indice, pas une preuve : rien ne dit que ce taux de victoire
se maintiendrait sur un idiome jamais vu, ni s'il resiste a un vrai jeu de
test.

Ici, on formalise ca comme un vrai probleme de classification binaire
supervisee :
  - une OBSERVATION = une paire (idiome candidat, texte) ;
  - une ETIQUETTE y in {0, 1} : y=1 si l'idiome est le bon pour ce texte,
    y=0 sinon (idiome tire au hasard parmi les 31 113 -- un "leurre") ;
  - une VARIABLE EXPLICATIVE (feature) par couche : Delta_l(i, t), pour
    l = 0 (couche d'embeddings brute) jusqu'a l = 24 (derniere couche de
    Qwen2.5-0.5B) ;
  - un MODELE : une regression logistique, qui apprend un seuil (et une
    pente) sur Delta_l pour predire y. C'est le classifieur le plus simple
    qui existe -- un choix deliberement minimal pour une etude EXPLORATOIRE :
    si meme ce modele-la separe bien les deux classes, le signal est solide ;
    s'il n'y arrive pas, un modele plus complexe ne sauverait probablement
    rien non plus (au stade exploratoire, la complexite du modele ne doit pas
    masquer l'absence de signal).

Pourquoi decouper l'evaluation PAR TEXTE et pas par ligne (piege classique) :
si les 1 + N_LEURRES lignes d'un meme texte pouvaient se retrouver a la fois
dans le train et dans le test, le modele pourrait "reconnaitre" indirectement
ce texte precis (par ex. son echelle de Delta_l generale) plutot que d'apprendre
une regle qui generalise a un texte JAMAIS vu. On force donc tout un texte
(l'idiome correct ET ses leurres) a rester entierement du meme cote de la
coupure train/test (sklearn.model_selection.GroupShuffleSplit, groupe =
identifiant du texte).

Pourquoi l'AIRE SOUS LA COURBE ROC (AUC) plutot que la seule exactitude
(accuracy) : il y a N_LEURRES fois plus d'exemples negatifs (y=0) que positifs
(y=1) dans le jeu de donnees (5 leurres pour 1 correct, ici). Un classifieur
qui ignore completement Delta_l et repond toujours "leurre" obtient deja
83.3% d'exactitude sans rien avoir appris -- ce chiffre seul ne dit rien. Une
maniere equivalente et plus parlante de definir l'AUC : c'est la probabilite
que, si l'on tire au hasard un exemple positif et un exemple negatif, le
classifieur attribue un score plus eleve au positif. AUC = 0.5 : autant dire
un tirage a pile ou face (aucun pouvoir separateur). AUC = 1.0 : separation
parfaite. Contrairement a l'exactitude, l'AUC ne depend pas du desequilibre
entre les deux classes -- c'est la mesure a lire en priorite ici."""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from chengyu.evaluation import charger_dico, normaliser, trouver_idiome
from chengyu.representation import modification_par_couche

FIGURE = "results/figures/09_auc_par_couche.png"

N_TEXTES = 100          # nombre de textes echantillonnes dans le corpus CIP
N_LEURRES = 5           # idiomes incorrects tires au hasard, PAR texte -- plus que
                        # le script 07 (1 seul) : il faut plusieurs exemples
                        # negatifs par texte pour qu'un classifieur ait quelque
                        # chose de substantiel a apprendre, pas juste une paire
TEST_SIZE = 0.3         # fraction des TEXTES (pas des lignes) reservee au test
SEED = 0                # graine : meme echantillon, mêmes leurres, meme decoupe
                        # a chaque execution -- resultats reproductibles

rng = random.Random(SEED)
dico, longueurs = charger_dico()
liste_idiomes = list(dico)
df = pd.read_csv("data/raw/cip/train.csv")

# --- Etape 1 : construire le jeu de donnees etiquete -----------------------
# Pour chaque texte du corpus CIP, on connait l'idiome qu'il paraphrase (la
# "cible") -- c'est notre seule source d'etiquettes y=1. Les etiquettes y=0
# sont fabriquees : on tire N_LEURRES idiomes au hasard dans le dictionnaire
# entier (31 113 entrees), en excluant la cible pour ne pas etiqueter par
# erreur un doublon comme "leurre".
lignes = []             # une ligne = une observation (texte, idiome, y, Delta_0..Delta_24)
n_couches = None        # nombre de couches de Qwen + 1 (embeddings) ; connu apres le 1er appel
evalues = sautes = 0

for texte_id, (src, dst) in enumerate(zip(df["src"], df["dst"])):
    if evalues >= N_TEXTES:
        break
    cible = trouver_idiome(src, dst, dico, longueurs)
    if cible is None:
        sautes += 1
        continue
    texte = normaliser(dst)

    leurres = rng.sample([i for i in liste_idiomes if i != cible], N_LEURRES)
    candidats = [(cible, 1)] + [(leurre, 0) for leurre in leurres]

    for idiome, etiquette in candidats:
        # modification_par_couche fait UN forward pass Qwen sur le prompt
        # "texte + idiome" (l'idiome doit venir apres le texte -- contrainte
        # causale, section 10 du chapitre) et renvoie Delta_l pour toutes les
        # couches en une seule fois (output_hidden_states=True ne coute rien
        # de plus qu'un forward pass normal).
        delta = modification_par_couche(idiome, texte)
        if n_couches is None:
            n_couches = len(delta)
        lignes.append({"texte_id": texte_id, "idiome": idiome, "y": etiquette,
                        **{f"delta_{l}": delta[l] for l in range(n_couches)}})

    evalues += 1
    print(f"[{evalues}/{N_TEXTES}] cible={cible}  ({N_LEURRES} leurres tires)")

print(f"\ntextes evalues : {evalues}  (sautes : {sautes})  "
      f"observations : {len(lignes)}  (1 correct + {N_LEURRES} leurres par texte)\n")

# --- Etape 2 : mettre en forme (X = variables explicatives, y = etiquette) --
data = pd.DataFrame(lignes)
colonnes_delta = [f"delta_{l}" for l in range(n_couches)]
X = data[colonnes_delta].to_numpy()   # forme (n_observations, n_couches)
y = data["y"].to_numpy()              # forme (n_observations,), valeurs dans {0,1}
groupes = data["texte_id"].to_numpy() # pour la coupure train/test PAR TEXTE

# --- Etape 3 : coupure train/test par groupe (texte), pas par ligne --------
decoupe = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_train, i_test = next(decoupe.split(X, y, groupes))
print(f"train : {len(i_train)} observations ({len(set(groupes[i_train]))} textes)  "
      f"test : {len(i_test)} observations ({len(set(groupes[i_test]))} textes)\n")

# Reference : un classifieur qui ignore tout et repond toujours "leurre"
# (la classe majoritaire, N_LEURRES contre 1) -- sert a rappeler que
# l'exactitude seule est trompeuse ici (voir docstring).
taux_base = 1 - y[i_test].mean()
print(f"reference (predire toujours y=0, \"leurre\") : exactitude = {taux_base:.1%}\n")

# --- Etape 4 : un classifieur PAR COUCHE, univarie --------------------------
# Pour chaque couche l, on ajuste une regression logistique n'utilisant QUE
# Delta_l comme variable explicative : p(y=1 | Delta_l) = sigmoide(a*Delta_l + b).
# Objectif : mesurer, couche par couche, le pouvoir separateur de Delta_l
# SEUL a cette couche -- c'est l'equivalent, en classification proprement
# evaluee (train/test), du "taux de victoire" du script 07.
print(f"{'couche':>7} {'AUC':>6} {'exactitude':>11}")
resultats_couche = []
for l in range(n_couches):
    clf = LogisticRegression()
    clf.fit(X[i_train, l:l+1], y[i_train])
    proba = clf.predict_proba(X[i_test, l:l+1])[:, 1]   # p(y=1) predite, sur le test
    auc = roc_auc_score(y[i_test], proba)
    acc = accuracy_score(y[i_test], clf.predict(X[i_test, l:l+1]))
    resultats_couche.append((l, auc, acc))
    print(f"{l:>7} {auc:>6.3f} {acc:>10.1%}")

meilleure_couche, meilleure_auc, _ = max(resultats_couche, key=lambda r: r[1])

# --- Etape 4bis : tracer AUC(couche) -- lire la courbe, pas juste le pic ---
# Une seule courbe, construite une seule fois, a partir de TOUTES les
# observations groupees (pas une courbe par texte -- section 12.1 du
# chapitre, "comment choisir la couche"). Le trait plein AUC(l) est ce qu'il
# faut regarder ; le point marque n'est qu'un repere, pas la seule chose a
# retenir -- une bosse large et stable sur plusieurs couches voisines est un
# signal plus credible qu'un pic isole.
couches = [l for l, _, _ in resultats_couche]
aucs = [a for _, a, _ in resultats_couche]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.5, color="#93a4b8", linestyle="--", linewidth=1,
           label="hasard (AUC = 0.5)")
ax.plot(couches, aucs, color="#1f5c8a", linewidth=2, marker="o",
        markersize=4, label="AUC par couche")
ax.scatter([meilleure_couche], [meilleure_auc], color="#1f5c8a", s=70,
           zorder=5, edgecolor="white", linewidth=1)
ax.annotate(f"couche {meilleure_couche}\nAUC={meilleure_auc:.3f}",
            (meilleure_couche, meilleure_auc), textcoords="offset points",
            xytext=(8, 10), fontsize=9, color="#1f5c8a")
ax.set_xlabel("couche l (0 = embeddings)")
ax.set_ylabel("AUC (test, groupe par texte)")
ax.set_title(f"Separabilite de Delta_l(i,t) par couche -- n={evalues} textes, "
             f"{N_LEURRES} leurres/texte")
ax.set_ylim(0.35, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE, dpi=150)
print(f"\ncourbe AUC(couche) enregistree : {FIGURE}")

# --- Etape 5 : un classifieur MULTIVARIE, toutes les couches a la fois -----
# Ici, p(y=1 | Delta_0, ..., Delta_24) = sigmoide(somme_l w_l*Delta_l + b) :
# la regression logistique apprend un poids w_l par couche. Si ce modele fait
# nettement mieux que la meilleure couche seule, le signal est REPARTI sur
# plusieurs couches (elles apportent une information complementaire, pas
# redondante) ; sinon, une seule couche porte l'essentiel du signal et les
# autres n'ajoutent que du bruit.
clf_multi = LogisticRegression(max_iter=1000)
clf_multi.fit(X[i_train], y[i_train])
proba_multi = clf_multi.predict_proba(X[i_test])[:, 1]
auc_multi = roc_auc_score(y[i_test], proba_multi)
acc_multi = accuracy_score(y[i_test], clf_multi.predict(X[i_test]))

print(f"\nmodele multivarie (les {n_couches} couches ensemble) : "
      f"AUC = {auc_multi:.3f}  exactitude = {acc_multi:.1%}")

# Les coefficients w_l du modele multivarie indiquent quelles couches pesent
# le plus dans sa decision -- un |w_l| grand signifie que Delta_l a cette
# couche influence fortement la prediction (une fois les autres couches
# prises en compte), pas seulement pris isolement comme a l'etape 4.
coeffs = sorted(enumerate(clf_multi.coef_[0]), key=lambda c: -abs(c[1]))[:5]
print("couches les plus influentes dans le modele multivarie (|poids| le plus grand) :",
      ", ".join(f"couche {l} (poids={c:+.2f})" for l, c in coeffs))

print(f"""
Lecture :
- reference {taux_base:.1%} : l'exactitude d'un classifieur qui n'apprend RIEN
  de Delta_l (il repond toujours "leurre"). Toute exactitude proche de ce
  chiffre signifie une absence de signal utile -- c'est l'AUC qu'il faut
  regarder en priorite (elle, ne depend pas du desequilibre 1 correct pour
  {N_LEURRES} leurres).
- AUC = 0.5 : Delta_l a cette couche ne distingue pas mieux le correct du
  leurre qu'un tirage au hasard. AUC = 1.0 : separation parfaite sur le jeu
  de test (jamais vu pendant l'entrainement).
- meilleure couche prise seule : couche {meilleure_couche} (AUC = {meilleure_auc:.3f}).
- Comparer AUC du modele multivarie ({auc_multi:.3f}) a celle de la meilleure
  couche seule ({meilleure_auc:.3f}) : si les deux sont proches, la couche
  {meilleure_couche} porte l'essentiel du signal a elle seule (bon candidat
  pour proposeur_par_texte, section 12.2 du chapitre) ; si le modele
  multivarie fait nettement mieux, plusieurs couches se completent et il
  faudrait reflechir a en combiner plusieurs plutot que d'en choisir une seule.
""")
