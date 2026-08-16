"""04_argmax_eval.py — full-dictionary evaluation (no multiple choice).
For each text: exact ranking of ALL idioms, top-1 / top-10,
with and without prior. Also reports the target's rank, and (with
prior) whether the posterior's length marginal P(length | t) favors
the target's true length -- see posterior_by_length in argmax.py.

Evaluated on data/raw/cip/in_domain/test.in.csv, a held-out split with
no overlap with train.csv -- the prior (data/idiom_freq.json) is built
entirely from train.csv (see 00_build_freq.py), so evaluating "with
prior" accuracy on train.csv rows would be in-sample for the prior.
Rows are shuffled before sampling: test.in.csv, like train.csv, is not
guaranteed to be in random order, and script 10 already hit a bug
where unshuffled file-order sampling repeated one target
disproportionately."""
import pandas as pd
from chengyu.evaluation import load_dictionary, normalize, find_idiom
from chengyu.argmax import exact_posterior, posterior_by_length, text_scores
from chengyu.prior import log_prior

N = 50    # full dictionary => expensive: start small
dictionary, lengths = load_dictionary()
idiom_list = sorted(dictionary)   # sorted, not list(): set order depends on
                        # per-process hash randomization (PYTHONHASHSEED)
df = (
    pd.read_csv("data/raw/cip/in_domain/test.in.csv")
    .drop_duplicates(subset=["src", "dst"])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

t1_without = t1_with = t10_without = t10_with = evaluated = skipped = 0
length_correct = 0
true_length_mass_sum = 0.0
for src, dst in zip(df["src"], df["dst"]):
    if evaluated >= N:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if target is None:
        skipped += 1
        continue
    text = normalize(dst)
    raw_scores = text_scores(text, idiom_list)
    with_prior_scores = {i: v + log_prior(i) for i, v in raw_scores.items()}
    rank_without = sorted(raw_scores, key=raw_scores.get, reverse=True).index(target) + 1
    rank_with = sorted(with_prior_scores, key=with_prior_scores.get, reverse=True).index(target) + 1
    t1_without += rank_without == 1; t10_without += rank_without <= 10
    t1_with += rank_with == 1; t10_with += rank_with <= 10

    length_probs = posterior_by_length(exact_posterior(with_prior_scores))
    true_length = len(target)
    predicted_length = max(length_probs, key=length_probs.get)
    length_correct += predicted_length == true_length
    true_length_mass_sum += length_probs[true_length]

    evaluated += 1
    print(f"[{evaluated}/{N}] target={target}  rank without/with prior: {rank_without}/{rank_with}  "
          f"length true/predicted: {true_length}/{predicted_length}  "
          f"P(true length|t)={length_probs[true_length]:.3f}")

print(f"\nevaluated: {evaluated} (skipped: {skipped})")
print(f"top-1  without prior: {t1_without/evaluated:.1%}   with prior: {t1_with/evaluated:.1%}")
print(f"top-10 without prior: {t10_without/evaluated:.1%}  with prior: {t10_with/evaluated:.1%}")
print(f"length marginal: predicted length matches true length in "
      f"{length_correct/evaluated:.1%} of texts; mean P(true length|t) = "
      f"{true_length_mass_sum/evaluated:.3f}")
