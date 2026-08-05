"""12_fit_text_state_stats.py — precompute per-coordinate mean/std of
layer-CHENGYU_LAYER text states over a sample of CIP texts, for
standardizing before cosine similarity in mcmc.text_proposer. This removes
the rogue-dimension hub effect confirmed empirically in
scripts/11_check_hub_idiom.py (Part 3): one idiom's dominant cosine
similarity to every text state collapsed entirely once standardized.

Saved to disk (data/text_state_stats_layer{L}.npz) and loaded once by
representation.load_text_state_stats, rather than recomputed per call:
each text state needs a real forward pass, so estimating this from many
texts is real GPU cost that should happen once, not on every proposer
build.

Requires CHENGYU_LAYER, matching scripts 09/10/11."""
import os
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import normalize
from chengyu.representation import text_state_by_layer

_layer_env = os.environ.get("CHENGYU_LAYER")
if _layer_env is None:
    raise SystemExit(
        "CHENGYU_LAYER is not set. Pick the layer validated by "
        "09_classification_delta.py's AUC-by-layer study, then run:\n"
        "  CHENGYU_LAYER=<n> python scripts/12_fit_text_state_stats.py"
    )
LAYER = int(_layer_env)

SEED = 0
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "300"))
OUT = f"data/text_state_stats_layer{LAYER}.npz"

rng = random.Random(SEED)
df = pd.read_csv("data/raw/cip/train.csv")
sample_rows = rng.sample(range(len(df)), min(N_TEXTS, len(df)))
texts = [normalize(df["dst"].iloc[k]) for k in sample_rows]

print(f"computing layer-{LAYER} text states for {len(texts)} texts "
      f"({len(texts)} forward passes)...")
states = np.stack([text_state_by_layer(t)[LAYER].cpu().numpy()
                    for t in texts])

mu, sigma = states.mean(axis=0), states.std(axis=0)
os.makedirs("data", exist_ok=True)
np.savez(OUT, mu=mu, sigma=sigma, n_texts=len(texts), layer=LAYER)
print(f"saved: {OUT}  (dim={mu.shape[0]}, "
      f"sigma min={sigma.min():.4f} max={sigma.max():.4f})")
