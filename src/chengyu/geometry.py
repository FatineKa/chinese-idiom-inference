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
    return vecs.mean(dim=0).float().numpy()


def embeddings(idioms: list) -> np.ndarray:
    return np.stack([idiom_embedding(i) for i in idioms])


def reduce(X: np.ndarray, p: int):
    """PCA down to p dimensions. Returns (X_reduced, pca); pca.explained_variance_ratio_
    gives the fraction of variance retained."""
    pca = PCA(n_components=p, random_state=0)
    return pca.fit_transform(X), pca
