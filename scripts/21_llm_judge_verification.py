"""21_llm_judge_verification.py -- for every text's TOP-1 predicted idiom
(same full-dictionary ranking as script 04), ask Qwen directly whether that
idiom actually fits, using scoring.yes_no_judgment. This runs strictly
AFTER the ranking is already computed -- it does not change how the top-1
prediction is picked, it only checks the picked answer afterward.

Motivation: the corpus's labeled target is not necessarily the only valid
idiom for a text (several idioms can fit a sentence reasonably well; the
corpus only records one). Right now, a top-1 prediction that differs from
the labeled target is simply counted as wrong. This script splits that
"wrong" bucket into two: cases where Qwen itself, asked directly, agrees
the predicted idiom doesn't fit (a genuine miss), and cases where Qwen
says it DOES fit anyway (a plausible alternative the corpus just didn't
happen to record). Run on ALL texts, not only the mismatches, so the
judge's reliability can also be checked on the texts where top-1 already
matches the label (it should mostly say "yes" there -- if it doesn't,
that is itself informative about how much to trust the judge)."""
import os

import matplotlib.pyplot as plt
import pandas as pd

from chengyu.argmax import text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.prior import log_prior
from chengyu.scoring import yes_no_judgment

N = 50    # matches script 04's default -- full-dictionary ranking is
          # expensive, start small
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)
df = (
    pd.read_csv("data/raw/cip/in_domain/test.in.csv")
    .drop_duplicates(subset=["src", "dst"])
    .sample(frac=1, random_state=42)   # same seed as script 04, so this
                        # can be pointed at the same texts if run at the
                        # same N
    .reset_index(drop=True)
)

rows = []
evaluated = skipped = 0
for src, dst in zip(df["src"], df["dst"]):
    if evaluated >= N:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        skipped += 1
        continue
    text = normalize(dst)

    raw_scores = text_scores(text, idiom_list)
    with_prior = {i: v + log_prior(i) for i, v in raw_scores.items()}
    top1 = max(with_prior, key=with_prior.get)
    match = top1 == target
    # likelihood-only top-1 -- free here, raw_scores is already computed
    # for the with-prior ranking above; lets both methods' "how often does
    # it actually find the reference idiom" rate be read side by side,
    # matching script 04's with/without-prior comparison.
    top1_without = max(raw_scores, key=raw_scores.get)
    match_without = top1_without == target

    judged_yes, log_p_yes, log_p_no = yes_no_judgment(text, top1)

    evaluated += 1
    print(f"[{evaluated}/{N}] target={target}  top1(with prior)={top1} match={match}  "
          f"top1(no prior)={top1_without} match={match_without}  "
          f"judge={'yes' if judged_yes else 'no'}  "
          f"(log p(yes)={log_p_yes:.2f}, log p(no)={log_p_no:.2f})")
    rows.append({"target": target, "top1": top1, "match": match,
                  "top1_without": top1_without, "match_without": match_without,
                  "judged_yes": judged_yes, "log_p_yes": log_p_yes, "log_p_no": log_p_no})

data = pd.DataFrame(rows)
print(f"\nevaluated: {evaluated} (skipped: {skipped})\n")

matches = data[data["match"]]
mismatches = data[~data["match"]]

print("how often each method's top-1 pick IS the corpus reference idiom:")
print(f"  likelihood only:      {data['match_without'].mean():.1%}")
print(f"  likelihood + prior:   {data['match'].mean():.1%}")
print(f"\njudge says 'yes, fits' overall (on the likelihood+prior pick): "
      f"{data['judged_yes'].mean():.1%}")

if len(matches):
    print(f"judge agreement when top1 == target (sanity check, "
          f"expect mostly 'yes'): {matches['judged_yes'].mean():.1%} "
          f"({len(matches)} texts)")
if len(mismatches):
    print(f"judge agreement when top1 != target (plausible-alternative "
          f"check): {mismatches['judged_yes'].mean():.1%} "
          f"({len(mismatches)} texts)")

adjusted_correct = matches.shape[0] + mismatches["judged_yes"].sum()
adjusted_rate = adjusted_correct / evaluated
print(f"\njudge-adjusted top-1 (labeled-correct OR judge says it fits "
      f"anyway): {adjusted_rate:.1%}")

# --- Figure: three bars -- likelihood-only accuracy, likelihood+prior
# accuracy, and likelihood+prior accuracy SPLIT into labeled-correct
# (solid) vs. judge-approved-alternative (hatched, stacked on top) -- shows
# not just the adjusted number but how much of it is "extra credit" from
# the judge, which the printed summary states separately but doesn't show
# as one comparable picture. ---------------------------------------------
rate_without = data["match_without"].mean()
rate_with = data["match"].mean()
bonus = adjusted_rate - rate_with   # judge-approved-alternative share only

fig, ax = plt.subplots(figsize=(6, 4.5))
bars_x = [0, 1, 2]
labels = ["likelihood\nonly", "likelihood\n+ prior", "likelihood + prior\n(judge-adjusted)"]

ax.bar(0, rate_without, width=0.6, color="#93a4b8")
ax.bar(1, rate_with, width=0.6, color="#1f5c8a")
ax.bar(2, rate_with, width=0.6, color="#1f5c8a", label="labeled-correct")
ax.bar(2, bonus, width=0.6, bottom=rate_with, color="#2f7d55", hatch="//",
       edgecolor="white", label="judge-approved alternative")

for x, rate in [(0, rate_without), (1, rate_with)]:
    ax.text(x, rate + 0.01, f"{rate:.1%}", ha="center", fontsize=10)
ax.text(2, rate_with + bonus + 0.01, f"{adjusted_rate:.1%}", ha="center", fontsize=10)

ax.set_xticks(bars_x)
ax.set_xticklabels(labels)
ax.set_ylabel("rate of finding/being judged as the correct idiom")
ax.set_title(f"Top-1 accuracy by method, with and without judge credit -- n={evaluated} texts")
ax.set_ylim(0, min(1.0, rate_with + bonus + 0.15))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dbe6f0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
figure = "results/figures/21_judge_comparison.png"
os.makedirs(os.path.dirname(figure), exist_ok=True)
fig.savefig(figure, dpi=150)
print(f"figure saved: {figure}")
