"""10_compare_proposers.py — does the informed proposer actually mix
faster than uniform? Runs both on the same subset, text, and step
budget, then compares their visit frequencies against the exact
posterior and their acceptance rates, aggregated over CHENGYU_N_TEXTS
texts -- a single text is not enough to tell whether the informed
proposer helps in general, only whether it happened to help (or not)
for that one example.

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
from chengyu.mcmc import metropolis_hastings, text_proposer, uniform_proposer
from chengyu.prior import log_prior
from chengyu.scoring import score_summary

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
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "10"))
VERBOSE = N_TEXTS == 1   # print the full per-idiom table only for a single text;
                          # with several texts, that much output per text is noise --
                          # the point becomes the aggregate, not any one example.


def acceptance_rate(trace):
    """Fraction of steps where the state actually changed -- a proxy for
    how often proposals were accepted (an accepted proposal that happens
    to re-propose the same idiom would be missed, but that is negligible
    on a 20-candidate subset)."""
    moves = sum(trace[i] != trace[i - 1] for i in range(1, len(trace)))
    return moves / (len(trace) - 1)


def total_variation(exact, counts, n):
    """Distance in [0,1] between the exact posterior and a chain's empirical
    visit frequencies after burn-in -- 0 means the chain's marginal exactly
    matches the target at this step budget, 1 means they share no mass.
    Unlike acceptance rate (a proxy for mixing speed), this measures the
    thing we actually care about: how close the chain has gotten to the
    true posterior in the steps it was given."""
    return 0.5 * sum(abs(exact[i] - counts[i] / n) for i in exact)


def compare(text, subset, n_steps=N_STEPS, verbose=VERBOSE):
    # log_weight = likelihood + prior; kept separate here (instead of just
    # calling mcmc.log_weight) so the breakdown can be printed -- the 19
    # non-target candidates are the most frequent idioms overall, so it's
    # worth seeing directly whether the prior or the likelihood is what
    # actually drives the exact posterior, rather than assuming.
    likelihood = {i: score_summary(text, i) for i in subset}
    prior = {i: log_prior(i) for i in subset}
    w = {i: math.exp(likelihood[i] + prior[i]) for i in subset}
    Z = sum(w.values())
    exact = {i: w[i] / Z for i in subset}      # EXACT posterior

    progress = max(1, n_steps // 10) if verbose else None

    uniform = uniform_proposer(subset)
    trace_u = metropolis_hastings(text, subset[0], uniform, n_steps,
                                   progress_every=progress)
    trace_u_burned = trace_u[n_steps // 10:]

    subset_embeddings = embeddings(subset)
    informed = text_proposer(subset, text, subset_embeddings, LAYER,
                              temperature=TEMPERATURE)
    trace_i = metropolis_hastings(text, subset[0], informed, n_steps,
                                   progress_every=progress)
    trace_i_burned = trace_i[n_steps // 10:]

    c_u = Counter(trace_u_burned)
    c_i = Counter(trace_i_burned)

    if verbose:
        print(f"\n{'idiom':<8} {'loglik':>9} {'logprior':>9} {'exact':>8} "
              f"{'uniform':>8} {'informed':>8}")
        for i in sorted(subset, key=lambda x: -exact[x]):
            print(f"{i:<8} {likelihood[i]:>9.2f} {prior[i]:>9.2f} {exact[i]:>8.3f} "
                  f"{c_u[i] / len(trace_u_burned):>8.3f} "
                  f"{c_i[i] / len(trace_i_burned):>8.3f}")

    # target = subset[0] by construction (see __main__). Compact, always-on
    # summary of *why* the top candidate beat the target -- likelihood or
    # prior -- without printing the full 20-row table for every text.
    target = subset[0]
    top = max(subset, key=lambda x: exact[x])
    gap_loglik = likelihood[top] - likelihood[target]
    gap_logprior = prior[top] - prior[target]

    return {
        "acc_uniform": acceptance_rate(trace_u),
        "acc_informed": acceptance_rate(trace_i),
        "tvd_uniform": total_variation(exact, c_u, len(trace_u_burned)),
        "tvd_informed": total_variation(exact, c_i, len(trace_i_burned)),
        "top": top,
        "target": target,
        "gap_loglik": gap_loglik,
        "gap_logprior": gap_logprior,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/cip/train.csv")
    dictionary, lengths = load_dictionary()

    # 20 candidates per text: its target + 19 frequent idioms (same subset
    # style as 02), frequency list shared across texts, target excluded so
    # the subset is always exactly 20.
    with open("data/idiom_freq.json", encoding="utf-8") as f:
        freq = json.load(f)
    frequent = sorted(freq, key=freq.get, reverse=True)

    results = []
    for src, dst in zip(df["src"], df["dst"]):
        if len(results) >= N_TEXTS:
            break
        target = find_idiom(src, dst, dictionary, lengths)
        if not target:
            continue
        text = normalize(dst)
        subset = [target] + [i for i in frequent if i != target][:19]

        print(f"\n=== text {len(results) + 1}/{N_TEXTS} -- target: {target} ===")
        metrics = compare(text, subset, n_steps=N_STEPS)
        print(f"acceptance -- uniform: {metrics['acc_uniform']:.1%}  "
              f"informed: {metrics['acc_informed']:.1%}  |  "
              f"TVD to exact -- uniform: {metrics['tvd_uniform']:.3f}  "
              f"informed: {metrics['tvd_informed']:.3f}")
        won = "target won" if metrics["top"] == metrics["target"] else \
              f"top={metrics['top']}"
        print(f"  {won}  |  gap vs target -- loglik: {metrics['gap_loglik']:+.2f}  "
              f"logprior: {metrics['gap_logprior']:+.2f}")
        results.append(metrics)

    print(f"\n=== aggregate over {len(results)} texts "
          f"(layer={LAYER}, T={TEMPERATURE}, n_steps={N_STEPS}) ===")
    for key in ("acc_uniform", "acc_informed", "tvd_uniform", "tvd_informed",
                "gap_loglik", "gap_logprior"):
        vals = [r[key] for r in results]
        print(f"{key:<14} mean = {sum(vals) / len(vals):.3f}")
    n_target_won = sum(1 for r in results if r["top"] == r["target"])
    print(f"target was top candidate in {n_target_won}/{len(results)} texts")
