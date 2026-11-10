"""neighbours.py — option B of the demo shortlist. Live demo only.

- Idea: similar sentences use similar idioms.
- Index: every CIP training sentence -> vector (small Chinese sentence model).
- New sentence:
  - find the most similar training sentences (nearest neighbours)
  - their idioms = candidates
- Can only find idioms that appear in the training sentences.
- combine(): mixes option A (shortlist.py) and option B.
- Thesis results do NOT use this file.
- Accuracy cost measured by scripts/31_shortlist_recall.py."""
import os

import numpy as np

EXAMPLES_PATH = os.environ.get("CHENGYU_EXAMPLES", "data/examples_index.npz")
ENCODER = os.environ.get("CHENGYU_ENCODER", "BAAI/bge-small-zh-v1.5")

_encoder = None   # (tokenizer, model), loaded on first use


def idioms_from_neighbours(similarity: np.ndarray, example_idioms: list, k: int) -> list:
    """Most similar examples first -> their idioms, no repeats, at most k."""
    out, seen = [], set()
    for j in np.argsort(-similarity):
        idiom = example_idioms[j]
        if idiom not in seen:
            seen.add(idiom); out.append(idiom)
            if len(out) == k:
                break
    return out


def combine(a: list, b: list, k: int) -> list:
    """Take from a and b in turn, no repeats, at most k."""
    out, seen = [], set()
    for pair in zip(a, b):
        for idiom in pair:
            if idiom not in seen and len(out) < k:
                seen.add(idiom); out.append(idiom)
    for idiom in a[len(b):] + b[len(a):]:   # leftovers if lengths differ
        if idiom not in seen and len(out) < k:
            seen.add(idiom); out.append(idiom)
    return out


def encode(texts: list, batch: int = 64) -> np.ndarray:
    """Sentences -> unit vectors (CLS token, as bge recommends)."""
    global _encoder
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = (AutoTokenizer.from_pretrained(ENCODER),
                    AutoModel.from_pretrained(ENCODER).to(device).eval())
    tok, model = _encoder
    out = []
    for i in range(0, len(texts), batch):
        x = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(model.device)
        with torch.no_grad():
            cls = model(**x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_examples(texts: list, idioms: list, path: str = EXAMPLES_PATH) -> None:
    """Save example vectors + their idioms. float16 = half the file size."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, idioms=np.array(idioms), vectors=encode(texts).astype(np.float16))


def load_examples(path: str = EXAMPLES_PATH) -> tuple:
    """-> (idiom of each example, vectors)"""
    data = np.load(path)
    return list(data["idioms"]), data["vectors"].astype(np.float32)


def neighbour_candidates(text: str, example_idioms: list, vectors: np.ndarray, k: int = 200) -> list:
    """Option B: k candidate idioms for `text`."""
    similarity = vectors @ encode([text])[0]   # unit vectors -> dot = cosine
    return idioms_from_neighbours(similarity, example_idioms, k)
