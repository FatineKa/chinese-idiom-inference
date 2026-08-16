"""13_classification_s.py — exploratory classification study for s_ell,
mirroring 09_classification_delta.py's methodology exactly (same dataset
construction, model, train/test split by text, AUC metric, permutation
control) but for the informed proposer's actual signal, standardized
(scripts/11_check_hub_idiom.py Part 3 confirmed standardization removes a
rogue-dimension hub artifact that would otherwise have contaminated this
study). This is assumption #1 from the proposal-distribution bridge
paragraph -- "does s_ell itself separate correct idioms from distractors"
-- flagged as unchecked from the start, and not testable cleanly until the
hub was fixed.

Question: does

    s_ell(i,t) = cos( standardize(e_static(i)), standardize(e_ctx(t, ell)) )

distinguish the correct idiom from other candidates, for a given text t,
at each layer ell -- and is ell=11 (the layer 09 validated for Delta_l,
reused for s_ell without its own check per that same open assumption)
actually a good layer for THIS signal?

Much cheaper than 09's Delta_l study: s_ell needs only ONE forward pass
PER TEXT (text_state_by_layer, reused for every candidate and every
layer), not one per candidate -- e_static(i) is a static embedding-table
lookup, no forward pass. So this script affords the same n as 09 for a
fraction of the GPU cost.

Requires scripts/12_fit_text_state_stats.py to have been run first (fits
every layer from one pass, so this only needs to happen once)."""
import os
import random
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.geometry import embeddings, static_embedding_stats
from chengyu.representation import load_text_state_stats, text_state_by_layer

STANDARDIZE = os.environ.get("CHENGYU_STANDARDIZE", "1") == "1"   # default on
                        # (the validated fix), set CHENGYU_STANDARDIZE=0 for
                        # a direct comparison against Delta_l's own study,
                        # which never used standardization either -- same
                        # numbers, different output files so the two never
                        # collide or corrupt each other's resume data.
_suffix = "" if STANDARDIZE else "_nostd"
FIGURE = f"results/figures/13_auc_par_couche_s{_suffix}.png"
DATA_OUT = f"results/outputs/13_s_scores{_suffix}.csv"
os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)

_n_texts_env = os.environ.get("CHENGYU_N_TEXTS")
N_TEXTS = int(_n_texts_env) if _n_texts_env else None   # None = every valid
                        # row of train.csv; set an integer for a quicker trial
N_DISTRACTORS = 5       # same as 09, for direct comparability
TEST_SIZE = 0.3
SEED = 0

rng = random.Random(SEED)
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)
# train.csv is not randomly ordered (long runs sharing the same target),
# so shuffle before taking the first N_TEXTS -- same fix as scripts 04/10.
df = (
    pd.read_csv("data/raw/cip/train.csv")
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

# --- Step 1: list the candidates to evaluate (no model call here) ---------
to_evaluate = []        # (text_id, idiom, text, y)
skipped = 0
for text_id, (src, dst) in enumerate(zip(df["src"], df["dst"])):
    # N_TEXTS counts VALID texts found so far, not raw rows scanned --
    # otherwise a run of many skipped (ambiguous-target) rows near the
    # start silently returns fewer evaluated texts than requested.
    if N_TEXTS is not None and (text_id - skipped) >= N_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        skipped += 1
        continue
    text = normalize(dst)
    distractors = rng.sample([i for i in idiom_list if i != target], N_DISTRACTORS)
    for idiom, label in [(target, 1)] + [(d, 0) for d in distractors]:
        to_evaluate.append((text_id, idiom, text, label))

evaluated = len(to_evaluate) // (1 + N_DISTRACTORS)
print(f"standardize={STANDARDIZE}  |  output: {DATA_OUT}")
print(f"texts evaluated: {evaluated}  (skipped: {skipped})  "
      f"observations to evaluate: {len(to_evaluate)}  "
      f"(1 correct + {N_DISTRACTORS} distractors per text)")

# --- Step 2: one forward pass PER TEXT, reused across its candidates ------
if STANDARDIZE:
    mu_static, sigma_static = static_embedding_stats()
else:
    mu_static, sigma_static = 0.0, 1.0   # no-op: (x-0)/1 = x, reuses the
                        # same code path instead of duplicating it
static_cache = {}   # idiom -> (standardized) static embedding, computed once


def get_static_std(idiom):
    if idiom not in static_cache:
        static_cache[idiom] = (embeddings([idiom])[0] - mu_static) / sigma_static
    return static_cache[idiom]


# Resume support: a run this long (full corpus, ~90k texts) is exactly the
# kind of job a Colab disconnect interrupts partway through. If DATA_OUT
# already has results (from an earlier, interrupted run), skip texts
# already complete instead of re-paying their forward-pass cost -- this
# only helps if results/ is NOT on the ephemeral local disk (symlink it to
# Drive, same as data/, or this file dies with the runtime anyway).
already_done = set()
header_written = False
if os.path.exists(DATA_OUT):
    prior = pd.read_csv(DATA_OUT)
    counts = prior.groupby("text_id").size()
    already_done = set(counts[counts == 1 + N_DISTRACTORS].index)
    header_written = True
    print(f"resuming: {len(already_done)} texts already complete in "
          f"{DATA_OUT}, skipping them")

n_layers = None
LAYERS = None            # layers actually used for s_ell -- excludes 0
text_state_stats = None  # {layer: (mu, sigma)}, loaded once
n_texts_done = 0
n_remaining = evaluated - len(already_done)

for text_id, group in groupby(to_evaluate, key=lambda row: row[0]):
    if text_id in already_done:
        continue
    group = list(group)
    text = group[0][2]
    all_states = text_state_by_layer(text).cpu().numpy()   # (n_layers+1, dim)
    if n_layers is None:
        n_layers = all_states.shape[0]
        # Layer 0 is excluded: text_state_by_layer reads the state at the
        # LAST token of the fixed prompt template ("...成语「"), and at
        # layer 0 (raw embedding, no context mixing) that token's embedding
        # is the same for every text regardless of content -- s_0(i,t)
        # would be a constant per idiom, carrying zero information about
        # the text, even unstandardized. Unlike Delta_0, whose baseline is
        # the IDIOM's own state (varies by candidate), s_ell's layer-0 text
        # state is structurally uninformative, not just weak.
        LAYERS = list(range(1, n_layers))
        if STANDARDIZE:
            text_state_stats = {l: load_text_state_stats(l) for l in LAYERS}
        else:
            text_state_stats = {l: (0.0, 1.0) for l in LAYERS}   # no-op

    text_states_std = {l: (all_states[l] - text_state_stats[l][0]) / text_state_stats[l][1]
                        for l in LAYERS}

    batch_rows = []
    for _, idiom, _text, label in group:
        v = get_static_std(idiom)
        s_by_layer = {
            l: float(v @ w / max(np.linalg.norm(v) * np.linalg.norm(w), 1e-12))
            for l, w in text_states_std.items()
        }
        batch_rows.append({"text_id": text_id, "idiom": idiom, "y": label,
                            **{f"s_{l}": s_by_layer[l] for l in LAYERS}})
    pd.DataFrame(batch_rows).to_csv(DATA_OUT, mode="a", header=not header_written,
                                     index=False)
    header_written = True
    n_texts_done += 1
    if n_texts_done % 50 == 0:
        print(f"[{n_texts_done}/{n_remaining} new, "
              f"{len(already_done) + n_texts_done}/{evaluated} total] "
              f"texts processed (saved to {DATA_OUT})")

# Reload from disk rather than using only this run's in-memory rows: a
# resumed run's earlier texts exist only in the file, not in this
# process's memory.
data = pd.read_csv(DATA_OUT)
# Defends against a disconnect landing mid-write for one text: that text
# wouldn't be in already_done (its row count wouldn't match), so it gets
# reprocessed and appended again -- keep only the latest write per
# (text_id, idiom) so a stale partial write never silently duplicates rows.
n_before = len(data)
# reset_index is required, not cosmetic: dropping rows leaves gaps in the
# original row labels, and code below (the permutation control) uses
# .index.to_numpy() to index directly into position-based numpy arrays
# (X, y, y_perm) built from this same DataFrame -- without resetting,
# those labels no longer match array positions once any row is dropped.
data = data.drop_duplicates(subset=["text_id", "idiom"], keep="last").reset_index(drop=True)
if len(data) != n_before:
    print(f"dropped {n_before - len(data)} duplicate rows from a partial "
          f"write before an interruption")
if LAYERS is None:
    # Every remaining text was already done -- the loop body never ran, so
    # LAYERS/n_layers were never set. Infer them from the file instead.
    LAYERS = sorted(int(c.split("_")[1]) for c in data.columns if c.startswith("s_"))
    n_layers = max(LAYERS) + 1
print(f"\nobservations: {len(data)}\n")

# --- Step 3: shape the data, then split into train / validation / test ----
# Three-way split, not two: picking the best-scoring layer out of ~30 is
# itself a form of model selection. Selecting AND reporting the "best" AUC
# on the same held-out set optimistically biases it (the max over many
# noisy layer estimates tends to beat any single layer's true AUC).
# Validation set: layer selection. Test set: one final, honestly-reported
# AUC for the selected layer, touched only once.
s_columns = [f"s_{l}" for l in LAYERS]
X = data[s_columns].to_numpy()
y = data["y"].to_numpy()
groups = data["text_id"].to_numpy()
layer_pos = {l: i for i, l in enumerate(LAYERS)}   # layer number -> X column,
                        # since LAYERS excludes 0 and X's columns follow LAYERS'
                        # order, not raw layer numbers

split_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_trainval, i_test = next(split_test.split(X, y, groups))
split_val = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_train_rel, i_val_rel = next(split_val.split(X[i_trainval], y[i_trainval], groups[i_trainval]))
i_train, i_val = i_trainval[i_train_rel], i_trainval[i_val_rel]

print(f"train: {len(i_train)} observations ({len(set(groups[i_train]))} texts)  "
      f"val: {len(i_val)} observations ({len(set(groups[i_val]))} texts)  "
      f"test: {len(i_test)} observations ({len(set(groups[i_test]))} texts)\n")

# --- Companion stat: win rate, on the test set only (so it's measured on
# the same held-out rows as the final reported AUC, not mixed with rows
# used for training or layer selection) -------------------------------
test_text_ids = set(groups[i_test])
data_test = data[data["text_id"].isin(test_text_ids)]
print(f"{'layer':>5} {'win rate (test)':>16}")
for l in LAYERS:
    col = f"s_{l}"
    wins = total = 0
    for _, g in data_test.groupby("text_id"):
        target_s = g.loc[g["y"] == 1, col].iloc[0]
        distractor_s = g.loc[g["y"] == 0, col]
        wins += int((target_s > distractor_s).sum())   # higher s = favored,
                        # unlike Delta_l where smaller was the win condition
        total += len(distractor_s)
    print(f"{l:>5} {wins / total:>16.3f}")
print()

baseline_rate = 1 - y[i_test].mean()
print(f"baseline (always predict y=0, \"distractor\"): accuracy = {baseline_rate:.1%}\n")

# --- Step 4: one classifier PER LAYER, univariate, selected on VALIDATION --
print(f"{'layer':>7} {'val AUC':>7} {'val acc':>11}")
layer_results = []
for l in LAYERS:
    p = layer_pos[l]
    clf = LogisticRegression()
    clf.fit(X[i_train, p:p + 1], y[i_train])
    proba = clf.predict_proba(X[i_val, p:p + 1])[:, 1]
    auc = roc_auc_score(y[i_val], proba)
    acc = accuracy_score(y[i_val], clf.predict(X[i_val, p:p + 1]))
    layer_results.append((l, auc, acc))
    print(f"{l:>7} {auc:>7.3f} {acc:>10.1%}")

best_layer, best_val_auc, _ = max(layer_results, key=lambda r: r[1])

# --- Step 4bis: ONE final, honestly-reported test AUC for the selected
# layer -- and the same for layer 11 (pre-specified, not chosen by search,
# so its test AUC needs no correction) for a fair comparison ------------
p_best = layer_pos[best_layer]
clf_best = LogisticRegression()
clf_best.fit(X[i_train, p_best:p_best + 1], y[i_train])
final_test_auc = roc_auc_score(
    y[i_test], clf_best.predict_proba(X[i_test, p_best:p_best + 1])[:, 1])
print(f"\nselected layer {best_layer} (by validation AUC = {best_val_auc:.3f}): "
      f"test AUC = {final_test_auc:.3f}")

if 11 in LAYERS:
    p11 = layer_pos[11]
    clf_11 = LogisticRegression()
    clf_11.fit(X[i_train, p11:p11 + 1], y[i_train])
    layer_11_test_auc = roc_auc_score(
        y[i_test], clf_11.predict_proba(X[i_test, p11:p11 + 1])[:, 1])
else:
    layer_11_test_auc = None

# --- Step 4ter: permutation-label control (Hewitt & Liang, 2019) ----------
# Layer selection tried len(LAYERS) candidates and kept the max: a single
# permutation cannot control for that. Run many permutations and compare
# the OBSERVED max validation AUC against the null DISTRIBUTION of the max
# (not many separate per-layer checks).
N_PERMUTATIONS = int(os.environ.get("CHENGYU_N_PERM", "20"))
perm_rng = random.Random(SEED + 1)


def permuted_labels(rng):
    y_perm = y.copy()
    for _, g in data.groupby("text_id"):
        idx = g.index.to_numpy()
        y_perm[idx] = 0
        y_perm[rng.choice(idx.tolist())] = 1
    return y_perm


max_perm_val_aucs = []
perm_aucs_for_plot = None   # one representative permutation, for the figure
for b in range(N_PERMUTATIONS):
    y_perm = permuted_labels(perm_rng)
    perm_aucs_b = []
    for l in LAYERS:
        p = layer_pos[l]
        clf = LogisticRegression()
        clf.fit(X[i_train, p:p + 1], y_perm[i_train])
        proba = clf.predict_proba(X[i_val, p:p + 1])[:, 1]
        perm_aucs_b.append(roc_auc_score(y_perm[i_val], proba))
    max_perm_val_aucs.append(max(perm_aucs_b))
    if b == 0:
        perm_aucs_for_plot = perm_aucs_b

p_value = (1 + sum(t >= best_val_auc for t in max_perm_val_aucs)) / (N_PERMUTATIONS + 1)
print(f"\nmax-statistic permutation test ({N_PERMUTATIONS} permutations): "
      f"observed max val AUC = {best_val_auc:.3f}, "
      f"p = {p_value:.3f} (fraction of permuted max-AUCs >= observed)")

# single permutation check on the FINAL test AUC -- no multiple-comparison
# issue here, since only one (already-selected) layer is being evaluated
y_perm_test = permuted_labels(random.Random(SEED + 2))
clf_best_perm = LogisticRegression()
clf_best_perm.fit(X[i_train, p_best:p_best + 1], y_perm_test[i_train])
final_test_perm_auc = roc_auc_score(
    y_perm_test[i_test], clf_best_perm.predict_proba(X[i_test, p_best:p_best + 1])[:, 1])

# --- Step 4quat: plot validation-AUC(layer), real vs. permuted-label control
layers = [l for l, _, _ in layer_results]
val_aucs = [a for _, a, _ in layer_results]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.5, color="#93a4b8", linestyle="--", linewidth=1,
           label="chance (AUC = 0.5)")
ax.plot(layers, val_aucs, color="#1f5c8a", linewidth=2, marker="o",
        markersize=4, label="validation AUC by layer (s_ell)")
ax.plot(layers, perm_aucs_for_plot, color="#c0783c", linewidth=1.5, marker="o",
        markersize=3, linestyle="--", alpha=0.85,
        label="permuted-label control (1 of B permutations)")
ax.scatter([best_layer], [best_val_auc], color="#1f5c8a", s=70,
           zorder=5, edgecolor="white", linewidth=1)
ax.annotate(f"layer {best_layer}\nval AUC={best_val_auc:.3f}\ntest AUC={final_test_auc:.3f}",
            (best_layer, best_val_auc), textcoords="offset points",
            xytext=(8, 10), fontsize=9, color="#1f5c8a")
ax.axvline(11, color="#9a9a9a", linestyle=":", linewidth=1,
           label="layer 11 (chosen for Delta_l)")
ax.set_xlabel("layer l (0 = embeddings)")
ax.set_ylabel("AUC (validation, grouped by text)")
ax.set_title(f"Separability of s_ell(i,t) by layer -- n={evaluated} texts, "
             f"{N_DISTRACTORS} distractors/text "
             f"({'standardized' if STANDARDIZE else 'not standardized'})")
ax.set_ylim(0.0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE, dpi=150)
print(f"\nAUC(layer) curve saved: {FIGURE}")

# --- Step 5: one MULTIVARIATE classifier, all layers at once --------------
# No layer selection here (a single model, not compared against
# alternatives to pick a "best" one), so train/test needs no validation set.
clf_multi = LogisticRegression(max_iter=1000)
clf_multi.fit(X[i_train], y[i_train])
proba_multi = clf_multi.predict_proba(X[i_test])[:, 1]
auc_multi = roc_auc_score(y[i_test], proba_multi)
acc_multi = accuracy_score(y[i_test], clf_multi.predict(X[i_test]))
print(f"\nmultivariate model (all {len(LAYERS)} layers together, layer 0 "
      f"excluded): AUC = {auc_multi:.3f}  accuracy = {acc_multi:.1%}")

# --- Step 6: permutation-label control task, multivariate model ----------
y_perm_multi = permuted_labels(random.Random(SEED + 3))
clf_multi_perm = LogisticRegression(max_iter=1000)
clf_multi_perm.fit(X[i_train], y_perm_multi[i_train])
proba_multi_perm = clf_multi_perm.predict_proba(X[i_test])[:, 1]
auc_multi_perm = roc_auc_score(y_perm_multi[i_test], proba_multi_perm)
print(f"\nmultivariate model on permuted labels: AUC = {auc_multi_perm:.3f}")

if layer_11_test_auc is not None:
    layer_11_line = f"test AUC = {layer_11_test_auc:.3f}."
    if final_test_auc - layer_11_test_auc < 0.02:
        layer_11_line += " Close to selected layer -> the layer transfer assumption holds."
    else:
        layer_11_line += (" Meaningfully below the selected layer -> the layer transfer "
                           "assumption does NOT hold, a different layer "
                           "should be used for s_ell.")
else:
    layer_11_line = "n/a (fewer than 12 layers)."

print(f"""
Summary:
- baseline {baseline_rate:.1%} (always "distractor"): any AUC close to
  0.5 means s_ell carries no usable signal at that layer.
- best single layer: {best_layer} (selected by validation AUC = {best_val_auc:.3f},
  honest held-out test AUC = {final_test_auc:.3f}).
- layer 11 (reused from Delta_l's study without its own check, per the
  bridge paragraph's second open assumption): {layer_11_line}
- multivariate AUC ({auc_multi:.3f}) vs. selected single layer ({final_test_auc:.3f}):
  close -> layer {best_layer} carries essentially all the signal;
  clearly higher -> several layers complement each other.
- layer-selection significance: p = {p_value:.3f} ({N_PERMUTATIONS} permutations,
  max-statistic test) -- probability that random noise alone would produce
  a best-of-{len(LAYERS)}-layers validation AUC at least this high.
- control task: final-layer test permuted-label AUC = {final_test_perm_auc:.3f},
  multivariate permuted-label AUC = {auc_multi_perm:.3f} (both should be
  close to 0.5). If either is as high as the corresponding real-label AUC,
  that is evidence of a leak in the pipeline, not of s_ell carrying
  information -- investigate before trusting the real AUC values.
""")
