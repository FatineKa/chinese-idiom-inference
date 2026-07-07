"""evaluation.py : normalisation du texte et extraction
   de l'idiome cible depuis une paire (src, dst) de CIP."""

def normaliser(texte: str) -> str:
    """Retire tous les espaces. train.csv est segmenté par espaces, pas les
       fichiers de test ni le chinois naturel -> on uniformise avant Qwen."""
    return "".join(str(texte).split())


def charger_dico(chemin="data/raw/cip/idioms.txt"):
    """Renvoie (ensemble des idiomes, liste des longueurs présentes).
       Les longueurs servent à repérer les idiomes par fenêtre glissante."""
    with open(chemin, encoding="utf-8") as f:
        dico = {l.strip() for l in f if l.strip()}
    longueurs = sorted({len(i) for i in dico}, reverse=True)
    return dico, longueurs


def idiomes_presents(texte_norm: str, dico: set, longueurs) -> set:
    """Tous les idiomes du dictionnaire présents dans le texte (déjà normalisé).
       Méthode : fenêtre glissante + test d'appartenance O(1) au set.
       Bien plus rapide que de tester les 31 113 idiomes un par un."""
    trouves = set()
    n = len(texte_norm)
    for L in longueurs:
        for k in range(n - L + 1):
            sub = texte_norm[k:k + L]
            if sub in dico:
                trouves.add(sub)
    return trouves


def trouver_idiome(src: str, dst: str, dico: set, longueurs):
    """L'idiome cible = présent dans src mais absent de dst (c'est celui que la
       paraphrase a déplié). Gère le faux ami 众所周知 (présent des deux côtés
       -> écarté). Renvoie None si le cas est ambigu (0 ou plusieurs candidats),
       pour garder un jeu d'évaluation propre."""
    s = normaliser(src) 
    d = normaliser(dst)
    candidats = idiomes_presents(s, dico, longueurs) - idiomes_presents(d, dico, longueurs)
    if len(candidats) == 1:
        return next(iter(candidats))
    return None                      # 0 candidat ou ambiguïté -> on saute la ligne