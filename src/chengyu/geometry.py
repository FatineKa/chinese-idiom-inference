"""geometry.py — idiom embeddings, first pillar of the thesis.

An idiom's vector is the average of the input embeddings (token table,
before any pass through the transformer's layers) of its characters. Used
by representation.py (the Delta_l signal, idiom-inference chapter)."""
import numpy as np
import torch
from sklearn.decomposition import PCA

from chengyu.scoring import _model, _tok


@torch.no_grad()
def idiom_embedding(idiom: str) -> np.ndarray:
    ids = _tok(idiom, add_special_tokens=False).input_ids
    vecs = _model.get_input_embeddings().weight[ids]
    return vecs.mean(dim=0).float().cpu().numpy()


def embeddings(idioms: list) -> np.ndarray:
    return np.stack([idiom_embedding(i) for i in idioms])


_static_stats_cache = None


def static_embedding_stats():
    """Per-coordinate mean and std of the full dictionary's static
    embeddings, for standardizing before cosine similarity -- removes the
    rogue-dimension effect (Timkey & van Schijndel, 2021) confirmed
    empirically in scripts/11_check_hub_idiom.py (Part 3: one idiom's
    dominant cosine similarity to every text state collapsed entirely once
    standardized). Cheap (embedding-table lookups, no forward pass), so
    computed directly here rather than precomputed to disk like the
    text-state stats (representation.py) -- cached in-process since every
    caller needs the same values."""
    global _static_stats_cache
    if _static_stats_cache is None:
        from chengyu.evaluation import load_dictionary
        dictionary, _ = load_dictionary()
        X = embeddings(sorted(dictionary))
        _static_stats_cache = (X.mean(axis=0), X.std(axis=0) + 1e-8)
    return _static_stats_cache


def reduce(X: np.ndarray, p: int):
    """PCA down to p dimensions. Returns (X_reduced, pca); pca.explained_variance_ratio_
    gives the fraction of variance retained."""
    pca = PCA(n_components=p, random_state=0)
    return pca.fit_transform(X), pca
