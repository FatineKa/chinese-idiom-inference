"""shortlist.py — fast candidate search. Live demo only.

- Problem: scoring 31,114 idioms with Qwen is too slow on a CPU.
- Step 1: keep the K idioms closest to the text (nearest neighbours).
  - text vector  = Qwen hidden state at one layer (1 forward pass)
  - idiom vector = static embedding (geometry.py), computed once
  - same signal as mcmc.text_proposer
- Step 2: exact posterior (argmax.py) on those K only.
- Thesis results do NOT use this file (full dictionary).
- Accuracy cost measured by scripts/31_shortlist_recall.py."""
import os

import numpy as np

INDEX_PATH = os.environ.get("CHENGYU_INDEX", "data/shortlist_index.npz")
LAYER = int(os.environ.get("CHENGYU_SHORTLIST_LAYER", "23"))  # pick with script 31


def cosine_to_all(vectors: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity: v vs every row of vectors."""
    norms = np.maximum(np.linalg.norm(vectors, axis=1) * np.linalg.norm(v), 1e-12)
    return vectors @ v / norms


def top_k(similarity: np.ndarray, idioms: list, k: int) -> list:
    """k most similar idioms, best first."""
    k = min(k, len(idioms))
    best = np.argpartition(-similarity, k - 1)[:k]
    return [idioms[j] for j in best[np.argsort(-similarity[best])]]


def build_index(idioms: list, path: str = INDEX_PATH) -> None:
    """Save all idiom vectors to disk. Table lookups only, no forward pass."""
    from chengyu.geometry import embeddings
    vectors = embeddings(idioms).astype(np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, idioms=np.array(idioms), vectors=vectors)


def load_index(path: str = INDEX_PATH) -> tuple:
    """-> (idioms, vectors)"""
    data = np.load(path)
    return list(data["idioms"]), data["vectors"]


def text_vector(text: str, layer: int = LAYER) -> np.ndarray:
    """Text hidden state at `layer`, just before where the idiom would start."""
    from chengyu.representation import text_state_by_layer
    return text_state_by_layer(text)[layer].cpu().numpy()


def shortlist(text: str, idioms: list, vectors: np.ndarray, k: int = 200, layer: int = LAYER) -> list:
    """Step 1 of the demo: k candidate idioms for `text`."""
    return top_k(cosine_to_all(vectors, text_vector(text, layer)), idioms, k)
