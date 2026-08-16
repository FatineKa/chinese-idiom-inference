import json
import os
from collections import Counter

import pandas as pd

from chengyu.argmax import exact_posterior
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.mcmc import log_weight, metropolis_hastings, uniform_proposer

N_STEPS = int(os.environ.get("CHENGYU_N_STEPS", "20000"))  # override for a quick test, e.g. CHENGYU_N_STEPS=500


def verify(text, subset, n_steps=N_STEPS):
    """Compares the exact posterior (computable on a small set) to the
    MCMC's visit frequencies. If they match, the MCMC is correct.
    Note: this "exact" posterior is exact only conditional on `subset` --
    it is not the posterior over the full dictionary, which is what
    chengyu.argmax.rank/exact_posterior compute."""
    log_weights = {i: log_weight(text, i) for i in subset}
    exact = exact_posterior(log_weights)       # EXACT posterior on `subset`,
                                                # via the log-sum-exp trick --
                                                # direct exp() can underflow
                                                # to 0 for long texts

    proposer_fn = uniform_proposer(subset)
    trace, accepted = metropolis_hastings(text, subset[0], proposer_fn, n_steps,
                                           progress_every=max(1, n_steps // 10))
    trace = trace[n_steps // 10:]              # burn-in
    c = Counter(trace)
    print(f"{'idiom':<8} {'exact':>8} {'MCMC':>8}")
    for i in sorted(subset, key=lambda x: -exact[x]):
        print(f"{i:<8} {exact[i]:>8.3f} {c[i] / len(trace):>8.3f}")

    tvd = 0.5 * sum(abs(exact[i] - c[i] / len(trace)) for i in exact)
    print(f"\nTVD (exact vs. MCMC, post-burn-in): {tvd:.4f}")
    print(f"acceptance rate: {sum(accepted) / len(accepted):.1%}")


if __name__ == "__main__":
    # 1. a clean text and its target
    df = pd.read_csv("data/raw/cip/train.csv")
    dictionary, lengths = load_dictionary()
    for src, dst in zip(df["src"], df["dst"]):
        target = find_idiom(src, dst, dictionary, lengths)
        if target:
            break
    text = normalize(dst)
    print("text  :", text)
    print("target:", target, "\n")

    # 2. 20 candidates: the target + 19 frequent idioms
    with open("data/idiom_freq.json", encoding="utf-8") as f:
        freq = json.load(f)
    frequent = sorted(freq, key=freq.get, reverse=True)
    subset = [target] + [i for i in frequent if i != target][:19]

    # 3. exact vs. MCMC
    verify(text, subset, n_steps=N_STEPS)
