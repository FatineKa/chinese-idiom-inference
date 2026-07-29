import json
import math
import os
from collections import Counter

import pandas as pd

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.mcmc import log_weight, metropolis_hastings, uniform_proposer

N_STEPS = int(os.environ.get("CHENGYU_N_STEPS", "20000"))  # override for a quick test, e.g. CHENGYU_N_STEPS=500


def verify(text, subset, n_steps=N_STEPS):
    """Compares the exact posterior (computable on a small set) to the
    MCMC's visit frequencies. If they match, the MCMC is correct."""
    w = {i: math.exp(log_weight(text, i)) for i in subset}
    Z = sum(w.values())
    exact = {i: w[i] / Z for i in subset}      # EXACT posterior

    proposer_fn = uniform_proposer(subset)
    trace = metropolis_hastings(text, subset[0], proposer_fn, n_steps,
                                 progress_every=max(1, n_steps // 10))
    trace = trace[n_steps // 10:]              # burn-in
    c = Counter(trace)
    print(f"{'idiom':<8} {'exact':>8} {'MCMC':>8}")
    for i in sorted(subset, key=lambda x: -exact[x]):
        print(f"{i:<8} {exact[i]:>8.3f} {c[i] / len(trace):>8.3f}")


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
    with open("data/freq_idiomes.json", encoding="utf-8") as f:
        freq = json.load(f)
    frequent = sorted(freq, key=freq.get, reverse=True)
    subset = [target] + [i for i in frequent if i != target][:19]

    # 3. exact vs. MCMC
    verify(text, subset, n_steps=N_STEPS)
