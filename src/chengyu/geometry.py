"""geometry.py — plongements des idiomes (embeddings), premier pilier du mémoire.

Le vecteur d'un idiome est la moyenne des embeddings d'entrée (table de tokens,
avant tout passage dans les couches du transformeur) de ses caractères. Utilisé
par gmm_select.py et scripts/06_gmm_embeddings_bound.py pour l'étude géométrique
par mélange gaussien du chapitre de sélection de modèle."""
import numpy as np
import torch
from sklearn.decomposition import PCA

from chengyu.scoring import _model, _tok


@torch.no_grad()
def plongement_idiome(idiome: str) -> np.ndarray:
    ids = _tok(idiome, add_special_tokens=False).input_ids
    vecs = _model.get_input_embeddings().weight[ids]
    return vecs.mean(dim=0).float().numpy()


def plongements(idiomes: list) -> np.ndarray:
    return np.stack([plongement_idiome(i) for i in idiomes])


def reduire(X: np.ndarray, p: int):
    """PCA vers p dimensions. Renvoie (X_reduit, pca) ; pca.explained_variance_ratio_
    donne la part de variance conservée."""
    pca = PCA(n_components=p, random_state=0)
    return pca.fit_transform(X), pca
