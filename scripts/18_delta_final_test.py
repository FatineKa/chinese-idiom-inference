"""18_delta_final_test.py — the FROZEN final test: report one honest
pairwise success rate and Top-1 rate for ONE pre-committed configuration
(metric, layer, direction), on texts nothing has ever been picked from.

This is the missing third stage of the pipeline:
    calibration (script 16)  -> estimate mu_l, sigma_l
    validation  (script 17)  -> explore all 3 metrics x every layer,
                                 pick metric*, layer*, direction* BY EYE
    final test  (this script) -> freeze that one pick, report ONE number

The configuration is NOT re-derived here -- it's read from
CHENGYU_FINAL_METRIC / CHENGYU_FINAL_LAYER / CHENGYU_FINAL_DIRECTION (or the
defaults below), set AFTER looking at script 17's validation figures.
Re-deriving direction or "best layer" from this script's own data would
silently reintroduce the exact same problem script 17's docstring warns
about: picking whichever answer looks best on the same data you then report
it on.

Deliberately computes and saves ONLY the one frozen (metric, layer) pair --
not the full 29-layer x 3-metric grid script 17 saves. This isn't just an
efficiency choice: keeping the full grid around would make it easy to
quietly re-explore this "final" test set for an even better layer later,
which would undo the point of freezing a choice before looking.

Test texts: data/raw/cip/in_domain/test.in.csv, an OFFICIAL held-out split
of this dataset (same src/dst format as train.csv) that neither script 16
nor script 17 has ever read -- genuinely untouched, no need to carve out or
track a boundary within train.csv itself."""
import os
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import (
    find_idiom, load_dictionary, normalize, sample_length_matched_distractors,
)
from chengyu.representation import (
    context_states_batch, cosine_delta, euclidean_delta,
    load_context_state_stats, standardized_delta,
)
from chengyu.scoring import MODEL

TEST_FILE = "data/raw/cip/in_domain/test.in.csv"
DATA_OUT = "results/outputs/18_final_test_deltas.csv"
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)

METRIC = os.environ.get("CHENGYU_FINAL_METRIC", "cos")          # cos/std/euc
LAYER = int(os.environ.get("CHENGYU_FINAL_LAYER", "11"))
DIRECTION = os.environ.get("CHENGYU_FINAL_DIRECTION", "bigger")  # bigger/smaller
assert METRIC in ("cos", "std", "euc"), f"unknown metric {METRIC!r}"
assert DIRECTION in ("bigger", "smaller"), f"unknown direction {DIRECTION!r}"
METRIC_LABEL = {"cos": "cosine", "std": "standardized cosine", "euc": "Euclidean"}

K_DISTRACTORS = int(os.environ.get("CHENGYU_K", "20"))
N_TEST_TEXTS = int(os.environ.get("CHENGYU_N_TEST", "999999"))  # test.in.csv
                        # is small and fixed-size (4999 rows) -- unlike
                        # train.csv, no risk of an accidental huge run, so
                        # the default is simply "process the whole file"
BATCH_SIZE = int(os.environ.get("CHENGYU_BATCH", "32"))

print(f"FROZEN configuration: metric={METRIC_LABEL[METRIC]}, layer={LAYER}, "
      f"direction={DIRECTION} (set via CHENGYU_FINAL_METRIC/LAYER/DIRECTION, "
      f"NOT derived from this run's own data)\n")

dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)
by_length = {}   # {length: sorted list of idioms} -- must match scripts 16/17
for i in idiom_list:
    by_length.setdefault(len(i), []).append(i)

if METRIC == "std":
    mu0, sigma0 = load_context_state_stats(0)
    mu_l, sigma_l = load_context_state_stats(LAYER)
    meta = np.load("data/context_state_stats_layer0.npz")
    if str(meta["model"]) != MODEL:
        raise RuntimeError(
            f"data/context_state_stats_layer0.npz was fit with model "
            f"{meta['model']!r}, but this run uses {MODEL!r} -- rerun "
            f"scripts/16_fit_context_state_stats.py")

df = pd.read_csv(TEST_FILE)

# --- Step 1: build the frozen test set -- 1 target + K length-matched
# distractors per text, same construction as script 17's validation set ----
rng = random.Random(0)   # only used for distractor sampling,
                        # not for choosing which texts to include (this
                        # whole file is already untouched, so no shuffle
                        # or skip is needed the way scripts 16/17 needed one)
to_evaluate = []   # (text_id, idiom, text, is_target)
text_id = 0
for src, dst in zip(df["src"], df["dst"]):
    if text_id >= N_TEST_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        continue
    text = normalize(dst)
    distractors = sample_length_matched_distractors(target, by_length, K_DISTRACTORS, rng)
    for idiom, is_target in [(target, True)] + [(d, False) for d in distractors]:
        to_evaluate.append((text_id, idiom, text, is_target))
    text_id += 1

n_texts = text_id
print(f"final test set: {n_texts} texts x (1 target + {K_DISTRACTORS} distractors) "
      f"= {len(to_evaluate)} (idiom, text) pairs, from {TEST_FILE}\n")

# --- Step 2: batch through the model, computing ONLY the frozen metric at
# the frozen layer (see module docstring for why not the full grid) --------
if os.path.exists(DATA_OUT):
    os.remove(DATA_OUT)

rows = []
header_written = False
for start in range(0, len(to_evaluate), BATCH_SIZE):
    batch = to_evaluate[start:start + BATCH_SIZE]
    pairs = [(idiom, text) for _, idiom, text, _ in batch]
    states = context_states_batch(pairs)   # (batch, n_layers, dim)

    batch_rows = []
    for k, (tid, idiom, _text, is_target) in enumerate(batch):
        e0 = states[k, 0]
        el = states[k, LAYER]
        if METRIC == "cos":
            delta = cosine_delta(e0, el)
        elif METRIC == "std":
            delta = standardized_delta(e0, el, mu0, sigma0, mu_l, sigma_l)
        else:
            delta = euclidean_delta(e0, el)
        batch_rows.append({"text_id": tid, "idiom": idiom, "is_target": is_target,
                            "delta": delta})
    rows.extend(batch_rows)
    pd.DataFrame(batch_rows).to_csv(DATA_OUT, mode="a", header=not header_written, index=False)
    header_written = True
    print(f"[{min(start + BATCH_SIZE, len(to_evaluate))}/{len(to_evaluate)}] pairs processed")

data = pd.DataFrame(rows)
assert len(data) == n_texts * (K_DISTRACTORS + 1)

# --- Step 3: pairwise success rate and Top-1 rate, using the FROZEN
# direction -- not re-derived from this data (see module docstring) --------
values = data["delta"].to_numpy().reshape(n_texts, K_DISTRACTORS + 1)
target_vals = values[:, 0]
distractor_vals = values[:, 1:]

wins = int((target_vals[:, None] > distractor_vals).sum())
ties = int((target_vals[:, None] == distractor_vals).sum())
total = distractor_vals.size
W = (wins + 0.5 * ties) / total
pairwise_rate = W if DIRECTION == "bigger" else 1 - W

best_val = values.max(axis=1, keepdims=True) if DIRECTION == "bigger" else values.min(axis=1, keepdims=True)
is_best = values == best_val
n_tied = is_best.sum(axis=1)
target_credit = np.where(is_best[:, 0], 1.0 / n_tied, 0.0)
top1_rate = float(target_credit.mean())

chance_top1 = 1 / (K_DISTRACTORS + 1)
print(f"""
=== FINAL, FROZEN RESULT (never used for picking metric/layer/direction) ===
configuration: {METRIC_LABEL[METRIC]}, layer {LAYER}, direction={DIRECTION}
test set: {n_texts} texts from {TEST_FILE}, {K_DISTRACTORS} distractors/text

pairwise success rate: {pairwise_rate:.1%}   (chance = 50%)
Top-1 rate:             {top1_rate:.1%}   (chance = {chance_top1:.1%})

raw W (P(Delta(target) > Delta(distractor))) = {W:.3f}
""")
