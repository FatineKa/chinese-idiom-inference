"""10_compare_proposers.py — does the informed proposer actually mix
faster than uniform? Runs both on the same subset, text, and step
budget, then compares their visit frequencies against the exact
posterior and their acceptance rates.

Requires CHENGYU_LAYER: the layer validated by 09_classification_delta.py.
There is no safe default here -- an unvalidated layer is not meaningfully
different from a random guess, so the script refuses to run without one."""
import json
import math
import os
from collections import Counter

import pandas as pd

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.geometry import embeddings
from chengyu.mcmc import (
    log_weight,
    metropolis_hastings,
    text_proposer,
    uniform_proposer,
)

_layer_env = os.environ.get("CHENGYU_LAYER")
if _layer_env is None:
    raise SystemExit(
        "CHENGYU_LAYER is not set. Pick the layer validated by "
        "09_classification_delta.py's AUC-by-layer study, then run:\n"
        "  CHENGYU_LAYER=<n> python scripts/10_compare_proposers.py"
    )
LAYER = int(_layer_env)

N_STEPS = int(os.environ.get("CHENGYU_N_STEPS", "20000"))
TEMPERATURE = float(os.environ.get("CHENGYU_TEMPERATURE", "1.0"))


def acceptance_rate(trace):
    """Fraction of steps where the state actually changed -- a proxy for
    how often proposals were accepted (an accepted proposal that happens
    to re-propose the same idiom would be missed, but that is negligible
    on a 20-candidate subset)."""
    moves = sum(trace[i] != trace[i - 1] for i in range(1, len(trace)))
    return moves / (len(trace) - 1)


def compare(text, subset, n_steps=N_STEPS):
    w = {i: math.exp(log_weight(text, i)) for i in subset}
    Z = sum(w.values())
    exact = {i: w[i] / Z for i in subset}      # EXACT posterior

    uniform = uniform_proposer(subset)
    trace_u = metropolis_hastings(text, subset[0], uniform, n_steps,
                                   progress_every=max(1, n_steps // 10))
    trace_u_burned = trace_u[n_steps // 10:]

    subset_embeddings = embeddings(subset)
    informed = text_proposer(subset, text, subset_embeddings, LAYER,
                              temperature=TEMPERATURE)
    trace_i = metropolis_hastings(text, subset[0], informed, n_steps,
                                   progress_every=max(1, n_steps // 10))
    trace_i_burned = trace_i[n_steps // 10:]

    c_u = Counter(trace_u_burned)
    c_i = Counter(trace_i_burned)
    print(f"\n{'idiom':<8} {'exact':>8} {'uniform':>8} {'informed':>8}")
    for i in sorted(subset, key=lambda x: -exact[x]):
        print(f"{i:<8} {exact[i]:>8.3f} {c_u[i] / len(trace_u_burned):>8.3f} "
              f"{c_i[i] / len(trace_i_burned):>8.3f}")

    print(f"\nacceptance rate -- uniform: {acceptance_rate(trace_u):.1%}  "
          f"informed: {acceptance_rate(trace_i):.1%}")


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

    # 2. 20 candidates: the target + 19 frequent idioms (same subset as 02)
    with open("data/idiom_freq.json", encoding="utf-8") as f:
        freq = json.load(f)
    frequent = sorted(freq, key=freq.get, reverse=True)
    subset = [target] + [i for i in frequent if i != target][:19]

    # 3. uniform vs informed, same text, same subset, same step budget
    compare(text, subset, n_steps=N_STEPS)
