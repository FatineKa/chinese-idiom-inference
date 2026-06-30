# Projet mémoire — Inférence d'idiomes chinois (chengyu)

Inférer le chengyu (成语) qui résume un texte, vu comme une inférence
bayésienne `p(idiome | texte) = p(texte | idiome) · p(idiome) / Z`, où
Qwen fournit la vraisemblance et le prior vient des fréquences d'idiomes.

**Deux piliers :** l'étude géométrique des embeddings, et la sélection de
modèle (théorie de l'apprentissage). 

## Installation

```bash
conda activate projet-memoire        # ou : source .venv/bin/activate
pip install -e .
```

## Structure

- `src/chengyu/` — le code (un fichier par tâche) :
  `scoring.py`, `prior.py`, `embeddings.py`, `geometry.py`,
  `gmm_select.py`, `mcmc.py`, `evaluation.py`
- `scripts/` — les commandes à lancer (numérotées)
- `data/` — données brutes et nettoyées (non versionnées)
- `results/` — sorties (figures, tableaux)
- `config.yaml` — réglages (modèle, K, sigma, seed)

## Données

ChID (texte → bon idiome) :
`from datasets import load_dataset; load_dataset("thu-coai/chid")`

## Démarrage

```bash
python scripts/01_test_scoring.py    # vérifie que le bon idiome gagne
```

## Statut

En cours — mise en place du scoring et de l'étude des embeddings.