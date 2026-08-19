"""17_delta_controlled_test.py — controlled representation-study experiment
(chapter sections 5-11): does Delta_l(i,t) distinguish the correct idiom from
distractors, tested directly (pairwise success rate, Top-1 accuracy) instead
of only through a fitted classifier (script 09's AUC)?

For each held-out VALIDATION text: 1 target idiom + K distractors (the SAME
candidate set for every metric, so the three metrics are compared fairly).
Three distance metrics per candidate per layer:
  A. cosine        Delta_l^cos = 1 - cos(e^(0), e^(l))
  B. standardized cosine  (standardize coordinates first, then cosine)
  C. Euclidean      Delta_l^E  = ||e^(l) - e^(0)||_2
(representation.cosine_delta / standardized_delta / euclidean_delta, all
computed from ONE forward pass per candidate via context_states_batch --
metric B needs scripts/16_fit_context_state_stats.py to have been run first.)

Direction of the signal is not assumed: W_l = P(Delta_l(target) >
Delta_l(distractor)) is computed directly. W_l > 50% means larger Delta_l
tends to mean a better fit; W_l < 50% means smaller does. This is picked and
reported on the SAME held-out validation set -- deliberately so: this script
is an exploratory pass (does a directional signal exist at all, and where),
not a confirmatory claim about one selected layer/metric, so it does not
need the further validation/test split script 09 uses for that latter
purpose (see the representation-study design note in the chapter).

VALIDATION texts are a suffix of the same SEED-shuffled train.csv order used
by script 16: this script skips the first CHENGYU_N_CALIBRATION texts (must
match script 16's setting) before sampling its own texts, so the two sets
never share a text."""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.representation import (
    context_states_batch, cosine_delta, euclidean_delta,
    load_context_state_stats, standardized_delta,
)
from chengyu.scoring import MODEL

FIGURE_PAIRWISE = "results/figures/17_pairwise_rate_par_couche.png"
FIGURE_TOP1 = "results/figures/17_top1_rate_par_couche.png"
DATA_OUT = "results/outputs/17_deltas.csv"   # raw deltas per candidate, all
                        # 3 metrics x every layer -- saved before analysis so
                        # a re-run of the analysis alone never needs new
                        # (expensive) forward passes, same reasoning as
                        # script 09.
os.makedirs(os.path.dirname(FIGURE_PAIRWISE), exist_ok=True)
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)

SEED = 0                # must match scripts/16_fit_context_state_stats.py
N_CALIBRATION_SKIP = int(os.environ.get("CHENGYU_N_CALIBRATION", "300"))
                        # must match script 16's CHENGYU_N_CALIBRATION, or
                        # the two texts sets can overlap
N_VAL_TEXTS = int(os.environ.get("CHENGYU_N_VAL", "500"))
                        # deliberately a safe, finite default -- this stage
                        # is an exploratory pass (see module docstring), not
                        # the "run on everything" full-dataset study. To
                        # process every remaining valid text, pass an
                        # explicit large number (e.g. CHENGYU_N_VAL=200000):
                        # the loop below simply runs out of rows and stops,
                        # no special "unlimited" mode needed.
K_DISTRACTORS = int(os.environ.get("CHENGYU_K", "20"))
BATCH_SIZE = int(os.environ.get("CHENGYU_BATCH", "32"))

METRICS = ["cos", "std", "euc"]
METRIC_LABEL = {"cos": "cosine", "std": "standardized cosine", "euc": "Euclidean"}
METRIC_COLOR = {"cos": "#1f5c8a", "std": "#2f7d55", "euc": "#c0783c"}

rng = random.Random(SEED)
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)
df = pd.read_csv("data/raw/cip/train.csv")
df = (
    df[~df["dst"].map(normalize).duplicated()]   # dedupe by NORMALIZED text
                        # before splitting -- must match script 16's dedupe
                        # exactly, or the two texts sets can drift apart
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

# --- Step 1: skip the calibration texts, then take N_VAL_TEXTS validation
# texts, 1 target + K distractors each ------------------------------------
to_evaluate = []   # (text_id, idiom, text, is_target)
skip_found = 0
text_id = 0
for src, dst in zip(df["src"], df["dst"]):
    if text_id >= N_VAL_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        continue
    if skip_found < N_CALIBRATION_SKIP:
        skip_found += 1
        continue
    text = normalize(dst)
    distractors = rng.sample([i for i in idiom_list if i != target], K_DISTRACTORS)
    for idiom, is_target in [(target, True)] + [(d, False) for d in distractors]:
        to_evaluate.append((text_id, idiom, text, is_target))
    text_id += 1

n_texts = text_id
print(f"validation set: {n_texts} texts x (1 target + {K_DISTRACTORS} distractors) "
      f"= {len(to_evaluate)} (idiom, text) pairs "
      f"(skipped {skip_found} calibration texts)")

calib_stats = {}   # metric B's per-layer (mu, sigma), loaded lazily below
                    # once n_layers is known from the first batch (Step 2)

# --- Step 2: batch through the model, compute all 3 metrics x every layer,
# saved incrementally -------------------------------------------------------
if os.path.exists(DATA_OUT):
    os.remove(DATA_OUT)

rows = []
n_layers = None
header_written = False
for start in range(0, len(to_evaluate), BATCH_SIZE):
    batch = to_evaluate[start:start + BATCH_SIZE]
    pairs = [(idiom, text) for _, idiom, text, _ in batch]
    states = context_states_batch(pairs)   # (batch, n_layers, dim)
    if n_layers is None:
        n_layers = states.shape[1]
        calib_stats = {l: load_context_state_stats(l) for l in range(n_layers)}

        # sanity-check the calibration file's metadata (script 16) against
        # this run's own config, so a stale/mismatched file fails loudly
        # instead of silently standardizing with the wrong population
        meta = np.load("data/context_state_stats_layer0.npz")
        if str(meta["model"]) != MODEL:
            raise RuntimeError(
                f"data/context_state_stats_layer0.npz was fit with model "
                f"{meta['model']!r}, but this run uses {MODEL!r} -- rerun "
                f"scripts/16_fit_context_state_stats.py")
        if int(meta["seed"]) != SEED or int(meta["k_distractors"]) != K_DISTRACTORS:
            print(f"WARNING: calibration stats were fit with seed={int(meta['seed'])}, "
                  f"k_distractors={int(meta['k_distractors'])}, but this run uses "
                  f"SEED={SEED}, K_DISTRACTORS={K_DISTRACTORS} -- still valid, "
                  f"just not exactly the config scripts/16 was last run with")

    batch_rows = []
    for k, (tid, idiom, _text, is_target) in enumerate(batch):
        e0 = states[k, 0]
        mu0, sigma0 = calib_stats[0]
        row = {"text_id": tid, "idiom": idiom, "is_target": is_target}
        for l in range(n_layers):
            if l == 0:
                # l=0 is the baseline compared to itself -- exactly 0 by
                # definition for cosine and standardized cosine, forced
                # rather than computed to avoid float round-off residue
                # (same reasoning as representation.modification_by_layer).
                row[f"cos_{l}"] = 0.0
                row[f"std_{l}"] = 0.0
            else:
                el = states[k, l]
                mu_l, sigma_l = calib_stats[l]
                row[f"cos_{l}"] = cosine_delta(e0, el)
                row[f"std_{l}"] = standardized_delta(e0, el, mu0, sigma0, mu_l, sigma_l)
            row[f"euc_{l}"] = euclidean_delta(e0, states[k, l])
        batch_rows.append(row)

    rows.extend(batch_rows)
    pd.DataFrame(batch_rows).to_csv(DATA_OUT, mode="a", header=not header_written, index=False)
    header_written = True
    print(f"[{min(start + BATCH_SIZE, len(to_evaluate))}/{len(to_evaluate)}] "
          f"pairs processed (saved to {DATA_OUT})")

data = pd.DataFrame(rows)
print(f"\nobservations: {len(data)}  (n_layers={n_layers})\n")

# --- Step 3: for each metric x layer, direction (W_l), pairwise success
# rate, and Top-1 rate, all on this same validation set (exploratory pass --
# see module docstring for why no further split is needed here). Vectorized
# via reshape rather than a pandas groupby per (metric, layer): Step 1 built
# to_evaluate with exactly K_DISTRACTORS+1 rows per text, target first, and
# that order is preserved all the way into `data` -- so a column's values
# reshape directly into (n_texts, K_DISTRACTORS+1) with no grouping needed.
# Matters at full-dataset scale (CHENGYU_N_VAL unset, tens of thousands of
# texts), where a fresh groupby per (metric, layer) would be far slower. ---
assert len(data) == n_texts * (K_DISTRACTORS + 1), \
    "expected exactly K_DISTRACTORS+1 rows per validation text, in order"

results = {m: {"W": [], "direction": [], "pairwise": [], "top1": []} for m in METRICS}
chance_top1 = 1 / (K_DISTRACTORS + 1)

for metric in METRICS:
    for l in range(n_layers):
        col = f"{metric}_{l}"
        values = data[col].to_numpy().reshape(n_texts, K_DISTRACTORS + 1)
        target_vals = values[:, 0]           # column 0 = target (Step 1's order)
        distractor_vals = values[:, 1:]      # columns 1.. = distractors

        # pairwise: pool every (target, distractor) pair across all texts. A
        # tie counts as HALF a win (same convention AUC uses for tied
        # scores) -- without this, layer 0 (where every candidate is
        # EXACTLY tied at Delta=0 by definition, the baseline compared to
        # itself) would score a fake 100%, since "not a strict win" got
        # silently read as "a win for the opposite direction".
        wins = int((target_vals[:, None] > distractor_vals).sum())
        ties = int((target_vals[:, None] == distractor_vals).sum())
        total = distractor_vals.size
        W_l = (wins + 0.5 * ties) / total

        if W_l >= 0.5:
            direction, pairwise_rate = "bigger", W_l
        else:
            direction, pairwise_rate = "smaller", 1 - W_l

        # top-1: under the chosen direction, is the target (column 0) the
        # single best candidate among all K+1? When several candidates tie
        # for best, split credit evenly among them (expected accuracy under
        # random tie-breaking) instead of always crediting whichever column
        # happens to come first -- argmax/argmin would otherwise always
        # "pick" the target at layer 0, since it sits in column 0 and every
        # candidate is tied there.
        best_val = values.max(axis=1, keepdims=True) if direction == "bigger" else values.min(axis=1, keepdims=True)
        is_best = values == best_val
        n_tied = is_best.sum(axis=1)
        target_credit = np.where(is_best[:, 0], 1.0 / n_tied, 0.0)
        top1_rate = float(target_credit.mean())

        results[metric]["W"].append(W_l)
        results[metric]["direction"].append(direction)
        results[metric]["pairwise"].append(pairwise_rate)
        results[metric]["top1"].append(top1_rate)

    print(f"{METRIC_LABEL[metric]:>22}: best layer by pairwise rate = "
          f"{int(np.argmax(results[metric]['pairwise']))} "
          f"({max(results[metric]['pairwise']):.3f}), "
          f"best layer by top-1 rate = "
          f"{int(np.argmax(results[metric]['top1']))} "
          f"({max(results[metric]['top1']):.3f})")

layers = list(range(n_layers))

# --- Step 4: two figures, one line per metric ------------------------------
# Figure 1 plots the RAW W_l = P(Delta_l(target) > Delta_l(distractor)), not
# the direction-corrected pairwise_rate = max(W_l, 1-W_l): picking whichever
# direction looks better and THEN plotting that number guarantees a curve
# that never dips below 50%, even for pure noise (max(W, 1-W) >= 0.5
# always) -- so it would hide exactly the information (a below-50% W_l means
# "smaller is better", not "no signal") this exploratory plot exists to show.
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.5, color="#93a4b8", linestyle="--", linewidth=1, label="no signal (50%)")
for metric in METRICS:
    ax.plot(layers, results[metric]["W"], color=METRIC_COLOR[metric],
             linewidth=2, marker="o", markersize=3, label=METRIC_LABEL[metric])
ax.set_xlabel("layer l (0 = embeddings)")
ax.set_ylabel("P(Delta(target) > Delta(distractor))  (validation)")
ax.set_title(f"Direction and strength of the Delta_l signal by layer -- "
             f"n={n_texts} texts, {K_DISTRACTORS} distractors/text")
ax.set_ylim(0.0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE_PAIRWISE, dpi=150)
print(f"\npairwise-rate figure saved: {FIGURE_PAIRWISE}")

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.axhline(chance_top1, color="#93a4b8", linestyle="--", linewidth=1,
            label=f"chance (1/{K_DISTRACTORS + 1} = {chance_top1:.1%})")
for metric in METRICS:
    ax2.plot(layers, results[metric]["top1"], color=METRIC_COLOR[metric],
              linewidth=2, marker="o", markersize=3, label=METRIC_LABEL[metric])
ax2.set_xlabel("layer l (0 = embeddings)")
ax2.set_ylabel("Top-1 rate (validation)")
ax2.set_title(f"Target is the single best candidate, by layer -- "
              f"n={n_texts} texts, {K_DISTRACTORS + 1} candidates/text")
ax2.set_ylim(0.0, 1.0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper right")
fig2.tight_layout()
fig2.savefig(FIGURE_TOP1, dpi=150)
print(f"top-1-rate figure saved: {FIGURE_TOP1}")

print("""
Summary: this is an EXPLORATORY pass (which layers/metrics carry a
directional signal, and which direction) on a single held-out validation
set. If a specific layer+metric combination is going to be reported as a
result (not just used to build intuition), re-confirm that one number on a
further held-out test set never touched here -- the same reasoning script 09
already applies to its own layer selection.
""")
