"""15_budget_comparison.py -- compares MCMC proposers (uniform vs the
edit-distance/length-aware mixture proposer, sec:character_validity)
against the EXACT posterior over the FULL dictionary, at an EQUAL total
number of model evaluations, not equal step count. Proposers have very
different per-step cost -- mixture_proposer's local branch scores a whole
edit-distance neighborhood per step (sometimes many idioms), uniform costs
exactly 1 model call/step -- so comparing at equal steps would silently
favor whichever proposer does more work per step.

Requires the real model (see chengyu/scoring.py); expensive (scores the
whole ~31k-idiom dictionary once for the exact posterior, per proposer's
own internal calls on top)."""
import os
from collections import Counter

import pandas as pd

from chengyu.argmax import exact_posterior, text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.mcmc import metropolis_hastings, mixture_proposer, uniform_proposer
from chengyu.prior import log_prior

BUDGET = int(os.environ.get("CHENGYU_BUDGET", "500"))   # total model evaluations per proposer
BATCH = int(os.environ.get("CHENGYU_BATCH", "10"))       # steps per budget-check
D = int(os.environ.get("CHENGYU_D", "1"))
EPSILON = float(os.environ.get("CHENGYU_EPSILON", "0.1"))
ALPHA = float(os.environ.get("CHENGYU_ALPHA", "0.5"))
SEED = 0

dictionary, lengths = load_dictionary()
idioms = sorted(dictionary)

# held-out, shuffled -- same reasoning as scripts 03/04 (the prior is
# built from train.csv, so evaluating on it would be in-sample for it)
df = (
    pd.read_csv("data/raw/cip/in_domain/test.in.csv")
    .drop_duplicates(subset=["src", "dst"])
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)
target = None
for src, dst in zip(df["src"], df["dst"]):
    target = find_idiom(src, dst, dictionary, lengths)
    if target:
        break
text = normalize(dst)
print("text  :", text)
print("target:", target, "\n")

# --- exact posterior over the FULL dictionary (ground truth) --------------
print("computing exact posterior over the full dictionary...")
raw = text_scores(text, idioms)
h = {i: raw[i] + log_prior(i) for i in idioms}
exact = exact_posterior(h)
rank = sorted(h, key=h.get, reverse=True).index(target) + 1
print(f"target posterior probability: {exact[target]:.4f}  (rank {rank}/{len(idioms)})\n")


def total_variation(exact, counts, n):
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / n) for k in keys)


def run_to_budget(make_proposer, name, budget, batch, seed):
    """Runs metropolis_hastings in batches, sharing one cache across
    batches, until `budget` distinct (text, idiom) model evaluations have
    been made -- not until a fixed number of steps."""
    cache = {}
    proposer = make_proposer(cache)
    current = target   # same start for every proposer, so differences
                        # reflect the proposal, not the initial state --
                        # same convention (and same limitation: it can
                        # hide poor burn-in) as scripts/10.
    full_trace = []
    step_seed = seed
    while len(cache) < budget:
        trace, _ = metropolis_hastings(text, current, proposer, batch,
                                        seed=step_seed, cache=cache)
        full_trace.extend(trace)
        current = trace[-1]
        step_seed += 1

    burn = max(1, len(full_trace) // 10)
    burned = full_trace[burn:]
    counts = Counter(burned)
    tvd = total_variation(exact, counts, len(burned))
    print(f"{name}: {len(full_trace)} steps, {len(cache)} model calls "
          f"(budget {budget}), TVD={tvd:.4f}, target visited "
          f"{counts[target]}/{len(burned)} times ({counts[target]/len(burned):.1%})")
    return tvd


run_to_budget(lambda cache: uniform_proposer(idioms), "uniform", BUDGET, BATCH, SEED)
run_to_budget(
    lambda cache: mixture_proposer(idioms, text, D=D, epsilon=EPSILON, alpha=ALPHA, cache=cache),
    f"mixture(D={D}, eps={EPSILON}, alpha={ALPHA})", BUDGET, BATCH, SEED)
