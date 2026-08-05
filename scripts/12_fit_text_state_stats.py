"""12_fit_text_state_stats.py — precompute per-coordinate mean/std of text
states at EVERY layer, over a sample of CIP texts, for standardizing
before cosine similarity (mcmc.text_proposer, scripts/13_classification_s.py).
This removes the rogue-dimension hub effect confirmed empirically in
scripts/11_check_hub_idiom.py (Part 3): one idiom's dominant cosine
similarity to every text state collapsed entirely once standardized.

One forward pass per text already returns every layer's state
(text_state_by_layer), so all layers' statistics are fit from the same
N_TEXTS forward passes -- no need to rerun this per layer, unlike an
earlier version of this script that required CHENGYU_LAYER.

Saved to disk, one file per layer (data/text_state_stats_layer{L}.npz),
and loaded once by representation.load_text_state_stats, rather than
recomputed per call: each text state needs a real forward pass, so
estimating this from many texts is real GPU cost that should happen once,
not on every proposer build."""
import os
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import normalize
from chengyu.representation import text_state_by_layer

SEED = 0
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "300"))

rng = random.Random(SEED)
df = pd.read_csv("data/raw/cip/train.csv")
sample_rows = rng.sample(range(len(df)), min(N_TEXTS, len(df)))
texts = [normalize(df["dst"].iloc[k]) for k in sample_rows]

print(f"computing text states at every layer for {len(texts)} texts "
      f"({len(texts)} forward passes, every layer read from each)...")
# text_state_by_layer(t): (n_layers+1, dim) -> stack over texts: (n_texts, n_layers+1, dim)
all_states = np.stack([text_state_by_layer(t).cpu().numpy() for t in texts])
n_layers = all_states.shape[1]

os.makedirs("data", exist_ok=True)
for layer in range(n_layers):
    states = all_states[:, layer, :]
    mu, sigma = states.mean(axis=0), states.std(axis=0)
    out = f"data/text_state_stats_layer{layer}.npz"
    np.savez(out, mu=mu, sigma=sigma, n_texts=len(texts), layer=layer)

print(f"saved stats for layers 0..{n_layers - 1} to "
      f"data/text_state_stats_layer*.npz  (dim={all_states.shape[2]})")
