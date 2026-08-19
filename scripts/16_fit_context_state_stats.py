"""16_fit_context_state_stats.py — precompute per-coordinate mean/std of the
IDIOM's own state (e^(0)..e^(L), representation.context_states_batch), at
EVERY layer, over a calibration set of (idiom, text) pairs. Used to
standardize Delta_l's cosine metric (metric B, section 6:
representation.standardized_delta / load_context_state_stats).

Unlike text_state_by_layer (used by script 12 for mcmc.text_proposer), which
is the TEXT's own state without any idiom, this fits stats for the IDIOM's
state after being placed in context -- the actual e^(0)/e^(l) quantities
Delta_l compares. Deliberately not geometry.static_embedding_stats() either:
that describes the idiom's characters averaged over the whole idiom, not the
last-token state extracted here. Fits every layer, including layer 0 (the
un-contextualized baseline), from the SAME forward passes used for every
other layer -- context_states_batch already returns all layers per pair, so
no extra model calls beyond the calibration pairs themselves.

Both target AND distractor idioms are included in the calibration
population, matching scripts/09_classification_delta.py's evaluation-set
shape (1 target + K distractors per text) -- fitting only on targets would
bias the normalization toward the positive class. Distractors are
length-matched to their target (sample_length_matched_distractors) -- the
dictionary is 95.4% length-4 idioms, so a uniform sample would almost always
make a non-length-4 target the odd one out by character count alone, a
shortcut unrelated to whether it fits the text.

Calibration texts are a fixed prefix of the same SEED-shuffled train.csv
order that scripts/17_delta_controlled_test.py uses; script 17 skips this
many rows before sampling its own validation texts, so the two sets never
overlap."""
import os
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import (
    find_idiom, load_dictionary, normalize, sample_length_matched_distractors,
)
from chengyu.representation import context_states_batch
from chengyu.scoring import MODEL

SEED = 0                # must match scripts/17_delta_controlled_test.py's
                        # shuffle seed, so the two texts sets are disjoint
                        # prefixes/suffixes of the same order, not overlapping
N_CALIBRATION_TEXTS = int(os.environ.get("CHENGYU_N_CALIBRATION", "300"))
K_DISTRACTORS = int(os.environ.get("CHENGYU_K", "20"))   # matches the
                        # controlled test's K, so the calibration population
                        # has the same target:distractor shape as validation
BATCH_SIZE = int(os.environ.get("CHENGYU_BATCH", "32"))

rng = random.Random(SEED)
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)   # sorted, not list(): set iteration order
                        # depends on PYTHONHASHSEED -- see script 09
by_length = {}   # {length: sorted list of idioms} -- for length-matched
                  # distractor sampling (representation-study length check)
for i in idiom_list:
    by_length.setdefault(len(i), []).append(i)
df = pd.read_csv("data/raw/cip/train.csv")
df = (
    df[~df["dst"].map(normalize).duplicated()]   # dedupe by NORMALIZED text
                        # before splitting -- otherwise the same underlying
                        # text (present under two raw rows) could end up on
                        # both the calibration and validation side, weakening
                        # the "separate population" claim behind standardization
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

# --- Step 1: take the first N_CALIBRATION_TEXTS valid texts, 1 target + K
# distractors each ---------------------------------------------------------
pairs = []   # (idiom, text)
found = 0
for src, dst in zip(df["src"], df["dst"]):
    if found >= N_CALIBRATION_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        continue
    text = normalize(dst)
    distractors = sample_length_matched_distractors(target, by_length, K_DISTRACTORS, rng)
    for idiom in [target] + distractors:
        pairs.append((idiom, text))
    found += 1

print(f"calibration set: {found} texts x (1 target + {K_DISTRACTORS} distractors) "
      f"= {len(pairs)} (idiom, text) pairs")

# --- Step 2: batch through the model, collecting raw per-layer states -----
all_states = []
for start in range(0, len(pairs), BATCH_SIZE):
    batch = pairs[start:start + BATCH_SIZE]
    all_states.append(context_states_batch(batch))
    print(f"[{min(start + BATCH_SIZE, len(pairs))}/{len(pairs)}] pairs processed")
all_states = np.concatenate(all_states, axis=0)   # (n_pairs, n_layers+1, dim)
n_layers = all_states.shape[1]

# --- Step 3: per-layer mean/std, saved to disk -----------------------------
os.makedirs("data", exist_ok=True)
for layer in range(n_layers):
    states = all_states[:, layer, :]
    mu = states.mean(axis=0)
    sigma = np.maximum(states.std(axis=0), 1e-8)   # floor: a coordinate with
                        # ~zero variance would otherwise blow up to inf/nan
                        # when used to standardize (representation.standardized_delta)
    np.savez(f"data/context_state_stats_layer{layer}.npz",
             mu=mu, sigma=sigma, n_pairs=len(pairs), layer=layer,
             seed=SEED, k_distractors=K_DISTRACTORS,
             n_calibration_texts=found, model=MODEL)
                        # metadata beyond mu/sigma so scripts/17 can catch a
                        # stale-file mismatch (e.g. a different Qwen
                        # checkpoint) instead of silently standardizing with
                        # the wrong population

print(f"saved stats for layers 0..{n_layers - 1} to "
      f"data/context_state_stats_layer*.npz  (dim={all_states.shape[2]})")
