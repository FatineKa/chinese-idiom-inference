"""03_cip_eval.py — 7-candidate multiple-choice evaluation.
   Input  = normalized dst (without the idiom -> no leakage).
   Target = idiom extracted by find_idiom.
   Compares accuracy with prior vs. without prior (likelihood only)."""

import random

import pandas as pd

from chengyu.argmax import text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.prior import log_prior

N = 200                 # number of examples evaluated
random.seed(0)

dictionary, lengths = load_dictionary()
idiom_list = list(dictionary)
df = pd.read_csv("data/raw/cip/train.csv")

correct_without, correct_with, evaluated, skipped = 0, 0, 0, 0

for src, dst in zip(df["src"], df["dst"]):
    if evaluated >= N:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:                     # ambiguous row -> skip
        skipped += 1
        continue

    text = normalize(dst)                  # input = paraphrase without the idiom
    # 7 candidates: the correct one + 6 drawn at random (distinct from it)
    distractors = random.sample([i for i in idiom_list if i != target], 6)
    candidates = distractors + [target]
    random.shuffle(candidates)

    # score of all 7 candidates in a single forward pass
    raw_scores = text_scores(text, candidates, batch_size=len(candidates))   # log p(text | i)
    # without prior: argmax of the likelihood alone
    pred_without = max(candidates, key=lambda i: raw_scores[i])
    # with prior: argmax of log w = likelihood + prior
    pred_with = max(candidates, key=lambda i: raw_scores[i] + log_prior(i))

    correct_without += (pred_without == target)
    correct_with += (pred_with == target)
    evaluated += 1
    if evaluated % 20 == 0:
        print(f"[{evaluated}/{N}] accuracy so far -- without prior: {correct_without/evaluated:.1%}  "
              f"with prior: {correct_with/evaluated:.1%}")

print(f"\nevaluated: {evaluated}  (rows skipped: {skipped})")
print(f"accuracy without prior: {correct_without}/{evaluated} = {correct_without/evaluated:.1%}")
print(f"accuracy with prior   : {correct_with}/{evaluated} = {correct_with/evaluated:.1%}")
