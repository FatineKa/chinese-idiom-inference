"""09_classification_delta.py — exploratory study BY CLASSIFICATION (advisor's
agreement, 2026-07-27), which replaces/completes script 07 to answer a
precise question: does the signal

    Delta_l(i, t) = 1 - cos( e_static(i), e_contextualized(i | t, layer l) )

(representation.py, section 11 of the chapter) let us DISTINGUISH the
correct idiom from other candidate idioms, for a given text t?

Script 07 already asked this question, but with a single comparison per
text (the correct one against ONE distractor) and a single descriptive
statistic (the "win rate": does the correct one have a smaller Delta_l
than the distractor, yes/no). That's a hint, not proof: nothing says this
win rate would hold on an idiom never seen, or whether it survives a real
test set.

Here, we formalize this as a genuine supervised binary classification
problem:
  - one OBSERVATION = one (candidate idiom, text) pair;
  - a LABEL y in {0, 1}: y=1 if the idiom is the correct one for this
    text, y=0 otherwise (an idiom drawn at random among the 31,113 -- a
    "distractor");
  - one EXPLANATORY VARIABLE (feature) per layer: Delta_l(i, t), for
    l = 0 (raw embedding layer) up to l = L (last layer of the loaded
    checkpoint -- L is never assumed in advance, it is read from the
    model);
  - a MODEL: a logistic regression, which learns a threshold (and a
    slope) on Delta_l to predict y. This is the simplest classifier that
    exists -- a deliberately minimal choice for an EXPLORATORY study:
    if even this model separates the two classes well, the signal is
    solid; if it can't, a more complex model probably wouldn't save
    anything either (at the exploratory stage, model complexity should
    not mask the absence of signal).

Why the evaluation is split BY TEXT and not by row (a classic trap): if
the 1 + N_DISTRACTORS rows of the same text could end up in both train
and test, the model could indirectly "recognize" that specific text (e.g.
its general Delta_l scale) rather than learning a rule that generalizes
to a text it has NEVER seen. We therefore force a whole text (the correct
idiom AND its distractors) to stay entirely on the same side of the
train/test split (sklearn.model_selection.GroupShuffleSplit, group =
text identifier).

Why the AREA UNDER THE ROC CURVE (AUC) rather than plain accuracy: there
are N_DISTRACTORS times more negative examples (y=0) than positive ones
(y=1) in the dataset (5 distractors for 1 correct idiom, here). A
classifier that completely ignores Delta_l and always answers
"distractor" already reaches 83.3% accuracy without having learned
anything -- that number alone says nothing. An equivalent and more
telling way to define AUC: it is the probability that, drawing one
positive and one negative example at random, the classifier assigns a
higher score to the positive one. AUC = 0.5: a coin flip (no separating
power at all). AUC = 1.0: perfect separation. Unlike accuracy, AUC does
not depend on the imbalance between the two classes -- it is the metric
to read first here."""
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

_n_texts_env = os.environ.get("CHENGYU_N_TEXTS")
N_TEXTS = int(_n_texts_env) if _n_texts_env else None
                        # None (default) = every valid row of train.csv (95,560 rows,
                        # some of which will be skipped for lack of an
                        # identifiable target idiom -- a data-quality
                        # filter, not a sampling choice). No reason to cap
                        # arbitrarily if the GPU keeps up: more texts
                        # improves both the classifier's fit AND the
                        # reliability of the measured test AUC (chapter
                        # section 12.1). Only set an integer here for a
                        # quick/time-limited trial -- first time a single
                        # call to modification_by_layer on the target GPU
                        # to estimate the total runtime before launching on
                        # everything.
N_DISTRACTORS = 5       # incorrect idioms drawn at random, PER text -- more
                        # than script 07 (just 1): a classifier needs
                        # several negative examples per text to have
                        # something substantial to learn from, not just a
                        # single pair
TEST_SIZE = 0.3         # fraction of TEXTS (not rows) held out for testing
SEED = 0                # seed: same sample, same distractors, same split
                        # on every run -- reproducible results
BATCH_SIZE = 32         # candidates processed in ONE forward pass (modification_by_layer_batch).
                        # On GPU, many small successive calls under-use the
                        # hardware's parallelism; grouping into batches is
                        # what makes processing the whole corpus practical
                        # in reasonable time. Adjust downward if GPU memory
                        # is limited, upward if it isn't.

rng = random.Random(SEED)
dictionary, lengths = load_dictionary()
idiom_list = list(dictionary)
df = pd.read_csv("data/raw/cip/train.csv")

# --- Step 1: list the candidates to evaluate (no model call here) ---------
# For each text in the CIP corpus, we know the idiom it paraphrases (the
# "target") -- our only source of y=1 labels. The y=0 labels are
# manufactured: we draw N_DISTRACTORS idioms at random from the whole
# dictionary (31,113 entries), excluding the target so we never mislabel a
# duplicate as a "distractor".
to_evaluate = []        # (text_id, idiom, text, y) -- everything that must go through the model
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

# --- Step 2: evaluate Delta_l in batches (batching -- section 12.1/14) ----
rows = []
n_layers = None
for start in range(0, len(to_evaluate), BATCH_SIZE):
    batch = to_evaluate[start:start + BATCH_SIZE]
    pairs = [(idiom, text) for _, idiom, text, _ in batch]
    deltas = modification_by_layer_batch(pairs)   # ONE forward pass for the whole batch
    if n_layers is None:
        n_layers = len(deltas[0])
    for (text_id, idiom, _text, label), delta in zip(batch, deltas):
        rows.append({"text_id": text_id, "idiom": idiom, "y": label,
                      **{f"delta_{l}": delta[l] for l in range(n_layers)}})
    print(f"[{min(start + BATCH_SIZE, len(to_evaluate))}/{len(to_evaluate)}] observations processed")

print(f"\nobservations: {len(rows)}\n")

# --- Step 2: shape the data (X = explanatory variables, y = label) --------
data = pd.DataFrame(rows)
delta_columns = [f"delta_{l}" for l in range(n_layers)]
X = data[delta_columns].to_numpy()    # shape (n_observations, n_layers)
y = data["y"].to_numpy()              # shape (n_observations,), values in {0,1}
groups = data["text_id"].to_numpy()   # for the train/test split BY TEXT

# --- Step 3: train/test split by group (text), not by row -----------------
split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_train, i_test = next(split.split(X, y, groups))
print(f"train: {len(i_train)} observations ({len(set(groups[i_train]))} texts)  "
      f"test: {len(i_test)} observations ({len(set(groups[i_test]))} texts)\n")

# Baseline: a classifier that ignores everything and always answers
# "distractor" (the majority class, N_DISTRACTORS to 1) -- a reminder that
# accuracy alone is misleading here (see docstring).
baseline_rate = 1 - y[i_test].mean()
print(f"baseline (always predict y=0, \"distractor\"): accuracy = {baseline_rate:.1%}\n")

# --- Step 4: one classifier PER LAYER, univariate ---------------------------
# For each layer l, we fit a logistic regression using ONLY Delta_l as the
# explanatory variable: p(y=1 | Delta_l) = sigmoid(a*Delta_l + b).
# Goal: measure, layer by layer, the separating power of Delta_l ALONE at
# that layer -- the properly evaluated (train/test) equivalent of script
# 07's "win rate".
print(f"{'layer':>7} {'AUC':>6} {'accuracy':>11}")
layer_results = []
for l in range(n_layers):
    clf = LogisticRegression()
    clf.fit(X[i_train, l:l+1], y[i_train])
    proba = clf.predict_proba(X[i_test, l:l+1])[:, 1]   # predicted p(y=1), on the test set
    auc = roc_auc_score(y[i_test], proba)
    acc = accuracy_score(y[i_test], clf.predict(X[i_test, l:l+1]))
    layer_results.append((l, auc, acc))
    print(f"{l:>7} {auc:>6.3f} {acc:>10.1%}")

best_layer, best_auc, _ = max(layer_results, key=lambda r: r[1])

# --- Step 4bis: plot AUC(layer) -- read the curve, not just the peak ------
# A single curve, built once, from ALL observations pooled together (not
# one curve per text -- chapter section 12.1, "how to choose the layer").
# The solid AUC(l) line is what matters; the marked point is only a
# landmark, not the only thing to remember -- a broad, stable bump across
# several neighboring layers is a more credible signal than an isolated
# spike.
layers = [l for l, _, _ in layer_results]
aucs = [a for _, a, _ in layer_results]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.5, color="#93a4b8", linestyle="--", linewidth=1,
           label="chance (AUC = 0.5)")
ax.plot(layers, aucs, color="#1f5c8a", linewidth=2, marker="o",
        markersize=4, label="AUC by layer")
ax.scatter([best_layer], [best_auc], color="#1f5c8a", s=70,
           zorder=5, edgecolor="white", linewidth=1)
ax.annotate(f"layer {best_layer}\nAUC={best_auc:.3f}",
            (best_layer, best_auc), textcoords="offset points",
            xytext=(8, 10), fontsize=9, color="#1f5c8a")
ax.set_xlabel("layer l (0 = embeddings)")
ax.set_ylabel("AUC (test, grouped by text)")
ax.set_title(f"Separability of Delta_l(i,t) by layer -- n={evaluated} texts, "
             f"{N_DISTRACTORS} distractors/text")
ax.set_ylim(0.0, 1.0)  # AUC is bounded in [0,1] -- a narrower limit
                        # was hiding real points below 0.35 on small samples
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE, dpi=150)
print(f"\nAUC(layer) curve saved: {FIGURE}")

# --- Step 5: one MULTIVARIATE classifier, all layers at once --------------
# Here, p(y=1 | Delta_0, ..., Delta_24) = sigmoid(sum_l w_l*Delta_l + b):
# the logistic regression learns one weight w_l per layer. If this model
# does clearly better than the best single layer, the signal is SPREAD
# across several layers (they contribute complementary, not redundant,
# information); otherwise, a single layer carries essentially all the
# signal and the others only add noise.
clf_multi = LogisticRegression(max_iter=1000)
clf_multi.fit(X[i_train], y[i_train])
proba_multi = clf_multi.predict_proba(X[i_test])[:, 1]
auc_multi = roc_auc_score(y[i_test], proba_multi)
acc_multi = accuracy_score(y[i_test], clf_multi.predict(X[i_test]))

print(f"\nmultivariate model (all {n_layers} layers together): "
      f"AUC = {auc_multi:.3f}  accuracy = {acc_multi:.1%}")

# The multivariate model's coefficients w_l indicate which layers weigh
# most in its decision -- a large |w_l| means Delta_l at that layer
# strongly influences the prediction (once the other layers are accounted
# for), not just taken in isolation as in step 4.
coeffs = sorted(enumerate(clf_multi.coef_[0]), key=lambda c: -abs(c[1]))[:5]
print("most influential layers in the multivariate model (largest |weight|):",
      ", ".join(f"layer {l} (weight={c:+.2f})" for l, c in coeffs))

print(f"""
Reading the results:
- baseline {baseline_rate:.1%}: the accuracy of a classifier that learns
  NOTHING from Delta_l (it always answers "distractor"). Any accuracy
  close to this number means no usable signal -- AUC is the metric to
  read first (it does not depend on the 1-correct-to-{N_DISTRACTORS}-distractors
  imbalance).
- AUC = 0.5: Delta_l at that layer distinguishes the correct idiom from
  the distractor no better than a coin flip. AUC = 1.0: perfect
  separation on the test set (never seen during training).
- best single layer: layer {best_layer} (AUC = {best_auc:.3f}).
- Compare the multivariate model's AUC ({auc_multi:.3f}) to the best single
  layer's ({best_auc:.3f}): if the two are close, layer {best_layer} carries
  essentially all the signal by itself (a good candidate for
  text_proposer, chapter section 12.2); if the multivariate model does
  clearly better, several layers complement each other and combining more
  than one should be considered instead of picking a single one.
""")
