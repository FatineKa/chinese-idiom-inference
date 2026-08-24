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

    judged_yes, log_p_yes, log_p_no = yes_no_judgment(text, top1)

    evaluated += 1
    print(f"[{evaluated}/{N}] target={target}  top1={top1}  "
          f"match={match}  judge={'yes' if judged_yes else 'no'}  "
          f"(log p(yes)={log_p_yes:.2f}, log p(no)={log_p_no:.2f})")
    rows.append({"target": target, "top1": top1, "match": match,
                  "judged_yes": judged_yes, "log_p_yes": log_p_yes, "log_p_no": log_p_no})

data = pd.DataFrame(rows)
print(f"\nevaluated: {evaluated} (skipped: {skipped})\n")

matches = data[data["match"]]
mismatches = data[~data["match"]]

print(f"top-1 accuracy (vs. labeled target): {data['match'].mean():.1%}")
print(f"judge says 'yes, fits' overall: {data['judged_yes'].mean():.1%}")

if len(matches):
    print(f"judge agreement when top1 == target (sanity check, "
          f"expect mostly 'yes'): {matches['judged_yes'].mean():.1%} "
          f"({len(matches)} texts)")
if len(mismatches):
    print(f"judge agreement when top1 != target (plausible-alternative "
          f"check): {mismatches['judged_yes'].mean():.1%} "
          f"({len(mismatches)} texts)")

adjusted_correct = matches.shape[0] + mismatches["judged_yes"].sum()
print(f"\njudge-adjusted top-1 (labeled-correct OR judge says it fits "
      f"anyway): {adjusted_correct / evaluated:.1%}")
