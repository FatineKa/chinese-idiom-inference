"""11_check_hub_idiom.py — is a candidate idiom's dominance in the informed
proposer (script 10) a real per-text signal, or a hub artifact?

Two independent checks, run against the SAME random sample of texts so
results are directly comparable, and each reported both raw and
per-text-centered (subtract that text's mean across all idioms tested).
Centering matters because both the informed proposer (a softmax over
candidates for one text) and the exact posterior (normalized over
candidates for one text) only ever compare idioms RELATIVE TO EACH OTHER
within a fixed text -- never in absolute terms across different texts. Raw
means mix in text-to-text variance (a longer or harder text shifts every
idiom's score together) that has nothing to do with any idiom being a hub;
centering removes it and isolates the idiom-specific effect.

Part 1 (embedding space): does e_static(idiom) sit in a generically
high-cosine-similarity direction relative to LAYER-CHENGYU_LAYER TEXT
STATES -- the actual comparison mcmc.py's text_proposer makes -- rather
than relative to other idioms' embeddings (a different, less relevant
space; an earlier version of this script tested that by mistake). One
forward pass per TEXT (not per idiom), since a text's state doesn't depend
on which candidate it gets compared against afterward.

Part 2 (likelihood space): does score_summary(text, idiom) run generically
higher for some idioms across many different, unrelated texts -- Qwen
treating them as broadly-applicable regardless of actual fit? One forward
pass per (text, idiom) pair, since score_summary requires inserting the
candidate into the prompt.

Requires CHENGYU_LAYER (Part 1 needs a depth to read the text state at;
use the layer validated by 09_classification_delta.py, same as script 10).
"""
import os
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import load_dictionary, normalize
from chengyu.geometry import embeddings
from chengyu.representation import text_state_by_layer
from chengyu.scoring import score_summary

_layer_env = os.environ.get("CHENGYU_LAYER")
if _layer_env is None:
    raise SystemExit(
        "CHENGYU_LAYER is not set. Pick the layer validated by "
        "09_classification_delta.py's AUC-by-layer study, then run:\n"
        "  CHENGYU_LAYER=<n> python scripts/11_check_hub_idiom.py"
    )
LAYER = int(_layer_env)

SEED = 0
N_TEXTS = 30
CANDIDATES = ["肆无忌惮", "无动于衷"]   # the two idioms that kept dominating
                        # in script 10's runs -- checked together so the
                        # "most favored idiom" isn't judged against nothing

rng = random.Random(SEED)
dictionary, _ = load_dictionary()
idiom_list = sorted(dictionary)
baseline_pool = [i for i in idiom_list if i not in CANDIDATES]   # exclude:
                        # sampling a CANDIDATES idiom into baseline too would
                        # duplicate a row and give it extra weight below
baseline = rng.sample(baseline_pool, 100)
idioms = CANDIDATES + baseline

df = pd.read_csv("data/raw/cip/train.csv")
sample_rows = rng.sample(range(len(df)), N_TEXTS)
texts = [normalize(df["dst"].iloc[k]) for k in sample_rows]

print(f"dictionary size: {len(idiom_list)}  |  layer: {LAYER}  |  "
      f"texts: {N_TEXTS}  |  idioms checked: {idioms}")


def cos(u, v):
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))


def report(name, matrix):
    """matrix[k, j] = idiom k's value against text j. Centering subtracts,
    per text (column), the mean across all idioms tested for that text --
    isolating the idiom-specific effect from text-to-text variance shared
    by every idiom, which raw mean/std cannot distinguish from a real
    hub effect."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    print(f"\n{name}")
    print(f"{'idiom':<10} {'raw mean':>10} {'raw std':>9} "
          f"{'centered mean':>14} {'centered std':>13}")
    for k, idiom in enumerate(idioms):
        tag = " <-- candidate" if idiom in CANDIDATES else ""
        print(f"{idiom:<10} {matrix[k].mean():>10.4f} {matrix[k].std():>9.4f} "
              f"{centered[k].mean():>14.4f} {centered[k].std():>13.4f}{tag}")


# --- Part 1: embedding space, against actual text states, not other idioms
print(f"\ncomputing layer-{LAYER} text states for {N_TEXTS} texts "
      f"({N_TEXTS} forward passes, one per text, reused for every idiom)...")
text_states = [text_state_by_layer(t)[LAYER].cpu().numpy() for t in texts]

static = {idiom: embeddings([idiom])[0] for idiom in idioms}
cos_matrix = np.array([[cos(static[idiom], v) for v in text_states]
                        for idiom in idioms])
report(f"Part 1: cos(e_static(idiom), text state at layer {LAYER}) "
       f"across {N_TEXTS} random texts", cos_matrix)

# --- Part 2: likelihood space, does score_summary run generically high? ---
print(f"\nscoring {len(idioms)} idioms x {N_TEXTS} texts = "
      f"{len(idioms) * N_TEXTS} forward passes...")
loglik_matrix = np.array([[score_summary(t, idiom) for t in texts]
                           for idiom in idioms])
report(f"Part 2: score_summary(text, idiom) across {N_TEXTS} random texts",
       loglik_matrix)

print(f"""
Reading this:
- 'raw mean'/'raw std': mixes in text-to-text variance shared by every
  idiom (a longer or harder text shifts everyone's score together), which
  has nothing to do with any one idiom being a hub -- this is what the
  first version of this script reported, and why Part 2's effect looked
  too small to trust against its own noise.
- 'centered mean'/'centered std': for each text, subtract that text's own
  mean across all {len(idioms)} idioms tested, then average. This is the
  quantity that actually matters -- both the informed proposer (a softmax
  over candidates for ONE text) and the exact posterior only ever compare
  idioms relative to each other within a fixed text, never in absolute
  terms across texts. A real hub has a HIGH centered mean AND a LOW
  centered std: consistently ranked above the pack, for reasons that don't
  depend on what the text says.
""")

# --- Part 3: does per-coordinate standardization kill the hub effect? -----
# Rogue dimensions (Timkey & van Schijndel, 2021) are a few coordinates with
# outsized variance across a whole population of vectors, dominating the dot
# product behind cosine similarity regardless of what the other coordinates
# encode. Standardizing each coordinate (subtract its mean, divide by its
# std) removes that -- computed separately for each side's own population,
# since static embeddings and layer-11 text states are different
# distributions: static side from the full 31,114-idiom dictionary (cheap,
# no forward pass), text-state side from the N_TEXTS sample already
# collected in Part 1 -- no extra forward passes needed for this part.
print(f"computing per-coordinate statistics for standardization "
      f"({len(idiom_list)} idioms, {N_TEXTS} text states)...")
X_all = embeddings(idiom_list)
mu_static, sigma_static = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
text_states_arr = np.stack(text_states)
mu_text = text_states_arr.mean(axis=0)
sigma_text = text_states_arr.std(axis=0) + 1e-8

static_std = {idiom: (static[idiom] - mu_static) / sigma_static
              for idiom in idioms}
text_states_std = [(v - mu_text) / sigma_text for v in text_states]

cos_matrix_std = np.array([[cos(static_std[idiom], v) for v in text_states_std]
                            for idiom in idioms])
report("Part 3: same comparison as Part 1, but with per-coordinate "
       "standardization applied first", cos_matrix_std)

print(f"""
Compare Part 3's centered mean/std for {CANDIDATES} against Part 1's:
- If standardization collapses the gap (candidates no longer clearly on
  top, relative to the baseline idioms), the dominance was a rogue-dimension
  artifact, and standardizing before computing s_ell would be a real fix
  for the informed proposer.
- If the gap survives standardization, something else is going on --
  standardization is not the fix, and the cause needs to be found elsewhere.
""")
