"""09_classification_delta.py — exploratory study BY CLASSIFICATION (advisor's
agreement, 2026-07-27), replacing/completing script 07. Question: does

    Delta_l(i, t) = 1 - cos( e_static(i), e_contextualized(i | t, layer l) )

(representation.py, chapter section 11) distinguish the correct idiom from
other candidates, for a given text t?

Script 07 asked this with a single distractor per text and a single
descriptive statistic (the win rate). That's a hint, not proof: it says
nothing about an idiom never seen, or about a held-out test set.

Formalized here as supervised binary classification:
  - OBSERVATION = one (candidate idiom, text) pair.
  - LABEL y=1 if the idiom is correct for this text, y=0 for a distractor
    drawn at random among the 31,113 idioms.
  - FEATURE per layer: Delta_l(i, t), l = 0 (raw embedding) to L (last
    layer, read from the model, never assumed in advance).
  - MODEL: logistic regression on Delta_l alone. Minimal by design: if
    this doesn't separate the classes, a more complex model won't save it
    either — at the exploratory stage, model complexity shouldn't mask an
    absent signal.

Split BY TEXT, not by row (GroupShuffleSplit, group = text id): if the
1+N_DISTRACTORS rows of one text could land on both sides of the split,
the model could recognize that text's general Delta_l scale instead of
learning a rule that generalizes to text it has never seen.

Metric: AUC, not accuracy. There are N_DISTRACTORS times more negative
than positive examples (5:1 here) — "always predict distractor" already
scores 83.3% accuracy while learning nothing, so accuracy alone is
uninformative. AUC = the probability that a random positive example is
scored higher than a random negative one (0.5 = chance, 1.0 = perfect
separation); it doesn't depend on the class imbalance, so it's the metric
to read first.
"""
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.representation import modification_by_layer_batch

FIGURE = "results/figures/09_auc_par_couche.png"
DATA_OUT = "results/outputs/09_deltas.csv"    # raw Delta_l per candidate, saved
                        # before any analysis so a re-run of this script or
                        # further checks (permutation control, rogue-dimension
                        # correction, ...) never require redoing the forward
                        # passes just to reuse data that already exists.
os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
# results/ is gitignored (not versioned), so a fresh clone doesn't have these
# directories at all -- create them rather than fail partway through a run.

_n_texts_env = os.environ.get("CHENGYU_N_TEXTS")
N_TEXTS = int(_n_texts_env) if _n_texts_env else None
                        # None = every valid row of train.csv (95,560 rows).
                        # More texts improves both the fit and the
                        # reliability of the measured AUC (chapter 12.1) --
                        # set an integer only for a quick/time-limited trial.
N_DISTRACTORS = 5       # distractors per text, more than script 07's 1: a
                        # classifier needs several negatives per text to
                        # learn from, not a single pair.
TEST_SIZE = 0.3         # fraction of TEXTS (not rows) held out for testing
SEED = 0                # fixed seed -- same sample and split on every run
BATCH_SIZE = 32         # candidates per forward pass (modification_by_layer_batch);
                        # batching is what makes the full corpus practical on GPU

rng = random.Random(SEED)
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)   # sorted, not list(): set iteration order
                        # depends on Python's per-process hash randomization
                        # (PYTHONHASHSEED), so a bare list() would make
                        # SEED-based reproducibility an illusion
df = pd.read_csv("data/raw/cip/train.csv")

# --- Step 1: list the candidates to evaluate (no model call here) ---------
to_evaluate = []        # (text_id, idiom, text, y)
skipped = 0

for text_id, (src, dst) in enumerate(zip(df["src"], df["dst"])):
    if N_TEXTS is not None and text_id >= N_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        skipped += 1
        continue
    text = normalize(dst)
    distractors = rng.sample([i for i in idiom_list if i != target], N_DISTRACTORS)
    for idiom, label in [(target, 1)] + [(distractor, 0) for distractor in distractors]:
        to_evaluate.append((text_id, idiom, text, label))

evaluated = len(to_evaluate) // (1 + N_DISTRACTORS)
print(f"texts evaluated: {evaluated}  (skipped: {skipped})  "
      f"observations to evaluate: {len(to_evaluate)}  (1 correct + {N_DISTRACTORS} distractors per text)")
print(f"processing in batches of {BATCH_SIZE} ({-(-len(to_evaluate)//BATCH_SIZE)} batches)\n")

# --- Step 2: evaluate Delta_l in batches, saved incrementally -------------
# Each batch is appended to DATA_OUT as soon as it's computed, not just held
# in memory and written once at the end: a long run on rented/paid compute
# can be interrupted (session timeout, disconnect, ...) partway through, and
# losing only the last few minutes of progress is a very different cost from
# losing the whole run. DATA_OUT is removed first so a previous run's rows
# are never silently appended onto a new one.
if os.path.exists(DATA_OUT):
    os.remove(DATA_OUT)

rows = []
n_layers = None
header_written = False
for start in range(0, len(to_evaluate), BATCH_SIZE):
    batch = to_evaluate[start:start + BATCH_SIZE]
    pairs = [(idiom, text) for _, idiom, text, _ in batch]
    deltas = modification_by_layer_batch(pairs)   # one forward pass for the whole batch
    if n_layers is None:
        n_layers = len(deltas[0])
    batch_rows = [
        {"text_id": text_id, "idiom": idiom, "y": label,
         **{f"delta_{l}": delta[l] for l in range(n_layers)}}
        for (text_id, idiom, _text, label), delta in zip(batch, deltas)
    ]
    rows.extend(batch_rows)
    pd.DataFrame(batch_rows).to_csv(DATA_OUT, mode="a", header=not header_written, index=False)
    header_written = True
    print(f"[{min(start + BATCH_SIZE, len(to_evaluate))}/{len(to_evaluate)}] observations processed "
          f"(saved to {DATA_OUT})")

print(f"\nobservations: {len(rows)}\n")

# --- Companion stat: win rate, the descriptive check AUC formalizes --------
# Per text/distractor: does the correct idiom have a smaller Delta_l than
# the distractor? No train/test split, no fitted model — purely descriptive.
data = pd.DataFrame(rows)

print(f"{'layer':>5} {'win rate':>10}")
for l in range(n_layers):
    col = f"delta_{l}"
    wins = total = 0
    for _, group in data.groupby("text_id"):
        target_delta = group.loc[group["y"] == 1, col].iloc[0]
        distractor_deltas = group.loc[group["y"] == 0, col]
        wins += int((target_delta < distractor_deltas).sum())
        total += len(distractor_deltas)
    print(f"{l:>5} {wins / total:>10.3f}")
print()

# --- Step 3: shape the data -------------------------------------------------
delta_columns = [f"delta_{l}" for l in range(n_layers)]
X = data[delta_columns].to_numpy()    # (n_observations, n_layers)
y = data["y"].to_numpy()
groups = data["text_id"].to_numpy()   # for the train/test split BY TEXT

split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_train, i_test = next(split.split(X, y, groups))
print(f"train: {len(i_train)} observations ({len(set(groups[i_train]))} texts)  "
      f"test: {len(i_test)} observations ({len(set(groups[i_test]))} texts)\n")

baseline_rate = 1 - y[i_test].mean()   # always predict y=0 ("distractor")
print(f"baseline (always predict y=0, \"distractor\"): accuracy = {baseline_rate:.1%}\n")

# --- Step 4: one classifier PER LAYER, univariate ---------------------------
# p(y=1 | Delta_l) = sigmoid(a*Delta_l + b), fit separately for each layer:
# the properly evaluated (train/test) equivalent of script 07's win rate.
print(f"{'layer':>7} {'AUC':>6} {'accuracy':>11}")
layer_results = []
for l in range(n_layers):
    clf = LogisticRegression()
    clf.fit(X[i_train, l:l+1], y[i_train])
    proba = clf.predict_proba(X[i_test, l:l+1])[:, 1]
    auc = roc_auc_score(y[i_test], proba)
    acc = accuracy_score(y[i_test], clf.predict(X[i_test, l:l+1]))
    layer_results.append((l, auc, acc))
    print(f"{l:>7} {auc:>6.3f} {acc:>10.1%}")

best_layer, best_auc, _ = max(layer_results, key=lambda r: r[1])

# --- Step 4bis: permutation-label control, per layer (Hewitt & Liang, 2019) -
# Same Delta_l values (X unchanged, no new forward passes) -- only the
# labels change: within each text's group of 1+N_DISTRACTORS candidates,
# the "correct" one is reassigned uniformly at random, independent of
# which candidate actually is correct. Delta_l can then carry no real
# information about y_perm by construction, so AUC here should collapse
# to ~0.5. If it doesn't, the pipeline (not Delta_l) is producing the
# apparent signal -- e.g. a leak in the grouped split -- and the AUC
# values measured above on the real labels cannot be trusted either.
perm_rng = random.Random(SEED + 1)
y_perm = y.copy()
for _, group in data.groupby("text_id"):
    idx = group.index.to_numpy()
    y_perm[idx] = 0
    y_perm[perm_rng.choice(idx.tolist())] = 1

print(f"\n{'layer':>7} {'perm AUC':>9}")
perm_aucs = []
for l in range(n_layers):
    clf = LogisticRegression()
    clf.fit(X[i_train, l:l+1], y_perm[i_train])
    proba = clf.predict_proba(X[i_test, l:l+1])[:, 1]
    auc = roc_auc_score(y_perm[i_test], proba)
    perm_aucs.append(auc)
    print(f"{l:>7} {auc:>9.3f}")

# --- Step 4ter: plot AUC(layer), real vs. permuted-label control ----------
# One curve over all observations pooled together. The full AUC(l) line is
# what matters; the marked point is a landmark, not the only thing to read
# — a broad, stable bump across neighboring layers is more credible than an
# isolated spike. The permuted curve should hug the chance line: if it
# doesn't, that undermines the real curve too (Step 6 below).
layers = [l for l, _, _ in layer_results]
aucs = [a for _, a, _ in layer_results]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.5, color="#93a4b8", linestyle="--", linewidth=1,
           label="chance (AUC = 0.5)")
ax.plot(layers, aucs, color="#1f5c8a", linewidth=2, marker="o",
        markersize=4, label="AUC by layer")
ax.plot(layers, perm_aucs, color="#c0783c", linewidth=1.5, marker="o",
        markersize=3, linestyle="--", alpha=0.85,
        label="permuted-label control")
ax.scatter([best_layer], [best_auc], color="#1f5c8a", s=70,
           zorder=5, edgecolor="white", linewidth=1)
ax.annotate(f"layer {best_layer}\nAUC={best_auc:.3f}",
            (best_layer, best_auc), textcoords="offset points",
            xytext=(8, 10), fontsize=9, color="#1f5c8a")
ax.set_xlabel("layer l (0 = embeddings)")
ax.set_ylabel("AUC (test, grouped by text)")
ax.set_title(f"Separability of Delta_l(i,t) by layer -- n={evaluated} texts, "
             f"{N_DISTRACTORS} distractors/text")
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
# p(y=1 | Delta_0..Delta_L) = sigmoid(sum_l w_l*Delta_l + b). If this beats
# the best single layer clearly, the signal is spread across layers
# (complementary, not redundant); otherwise one layer carries it essentially
# alone and the others only add noise.
clf_multi = LogisticRegression(max_iter=1000)
clf_multi.fit(X[i_train], y[i_train])
proba_multi = clf_multi.predict_proba(X[i_test])[:, 1]
auc_multi = roc_auc_score(y[i_test], proba_multi)
acc_multi = accuracy_score(y[i_test], clf_multi.predict(X[i_test]))

print(f"\nmultivariate model (all {n_layers} layers together): "
      f"AUC = {auc_multi:.3f}  accuracy = {acc_multi:.1%}")

coeffs = sorted(enumerate(clf_multi.coef_[0]), key=lambda c: -abs(c[1]))[:5]
print("most influential layers in the multivariate model (largest |weight|):",
      ", ".join(f"layer {l} (weight={c:+.2f})" for l, c in coeffs))

# --- Step 6: permutation-label control task, multivariate model ----------
# Reuses y_perm from Step 4bis (same random relabeling, one per text).
clf_multi_perm = LogisticRegression(max_iter=1000)
clf_multi_perm.fit(X[i_train], y_perm[i_train])
proba_multi_perm = clf_multi_perm.predict_proba(X[i_test])[:, 1]
auc_multi_perm = roc_auc_score(y_perm[i_test], proba_multi_perm)
best_perm_auc = max(perm_aucs)
print(f"\nmultivariate model on permuted labels: AUC = {auc_multi_perm:.3f}")
print(f"""
Control task reading: best permuted-label AUC = {best_perm_auc:.3f},
multivariate permuted-label AUC = {auc_multi_perm:.3f} (both should be close
to 0.5). If either is as high as the corresponding real-label AUC above,
that is evidence of a leak or bug in the evaluation pipeline, not of
Delta_l carrying information -- investigate before trusting the real
AUC values.
""")

print(f"""
Summary:
- baseline {baseline_rate:.1%} (always "distractor"): any AUC close to
  0.5 means Delta_l carries no usable signal at that layer.
- best single layer: {best_layer} (AUC = {best_auc:.3f}).
- multivariate AUC ({auc_multi:.3f}) vs. best single layer ({best_auc:.3f}):
  close -> layer {best_layer} carries essentially all the signal (candidate
  for text_proposer, chapter section 12.2); clearly higher -> several
  layers complement each other, consider combining rather than picking one.
""")
