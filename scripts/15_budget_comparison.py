"""15_budget_comparison.py -- compares MCMC proposers (uniform vs the
edit-distance/length-aware mixture proposer, sec:character_validity)
against the FULL dictionary, at an EQUAL total number of model evaluations
per text, not equal step count. Proposers have very different per-step
cost -- mixture_proposer's local branch scores a whole edit-distance
neighborhood per step (sometimes many idioms), uniform costs exactly 1
model call/step -- so comparing at equal steps would silently favor
whichever proposer does more work per step.

Aggregated over CHENGYU_N_TEXTS texts (default 50): a single text is an
anecdote, not evidence -- same reasoning as script 10.

Primary metric: TARGET VISIT RATE, the fraction of post-burn-in steps that
land exactly on the known-correct idiom. Cheap -- no model calls beyond the
MCMC budget already being spent -- and directly answers "does the informed
proposer actually find the right answer more often."

Optional, expensive secondary metric: TVD against the EXACT posterior over
the full ~31k-idiom dictionary (CHENGYU_EXACT_TVD=1). More complete, but one
exact posterior costs ~31,113 model calls (argmax.text_scores) -- N texts
therefore costs N times that, on top of the MCMC budgets themselves. Off by
default; only worth turning on once the target-visit-rate result looks
interesting enough to confirm more thoroughly.

Requires the real model (see chengyu/scoring.py)."""
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from chengyu.argmax import exact_posterior, text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.mcmc import metropolis_hastings, mixture_proposer, uniform_proposer
from chengyu.prior import log_prior

BUDGET = int(os.environ.get("CHENGYU_BUDGET", "500"))   # total model evaluations per proposer, per text
BATCH = int(os.environ.get("CHENGYU_BATCH", "10"))       # steps per budget-check
D = int(os.environ.get("CHENGYU_D", "1"))
EPSILON = float(os.environ.get("CHENGYU_EPSILON", "0.1"))
ALPHA = float(os.environ.get("CHENGYU_ALPHA", "0.5"))
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "50"))
EXACT_TVD = os.environ.get("CHENGYU_EXACT_TVD") == "1"
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


def run_to_budget(make_proposer, text, target, budget, batch, seed):
    """Runs metropolis_hastings in batches, sharing one cache across
    batches, until `budget` distinct (text, idiom) model evaluations have
    been made -- not until a fixed number of steps. Returns the post-burn-in
    trace and how many model calls it actually took."""
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
    return full_trace[burn:], len(cache)


results = []
for src, dst in zip(df["src"], df["dst"]):
    if len(results) >= N_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if not target:
        continue
    text = normalize(dst)

    burned_u, calls_u = run_to_budget(
        lambda cache: uniform_proposer(idioms), text, target, BUDGET, BATCH, SEED)
    burned_m, calls_m = run_to_budget(
        lambda cache: mixture_proposer(idioms, text, D=D, epsilon=EPSILON, alpha=ALPHA, cache=cache),
        text, target, BUDGET, BATCH, SEED)

    rate_u = burned_u.count(target) / len(burned_u)
    rate_m = burned_m.count(target) / len(burned_m)
    row = {"target": target, "rate_uniform": rate_u, "rate_mixture": rate_m,
           "calls_uniform": calls_u, "calls_mixture": calls_m}

    line = (f"[{len(results) + 1}/{N_TEXTS}] target={target}  "
            f"target-visit-rate -- uniform: {rate_u:.1%}  mixture: {rate_m:.1%}")

    if EXACT_TVD:
        raw = text_scores(text, idioms)
        h = {i: raw[i] + log_prior(i) for i in idioms}
        exact = exact_posterior(h)
        c_u, c_m = Counter(burned_u), Counter(burned_m)
        keys_u = set(exact) | set(c_u)
        keys_m = set(exact) | set(c_m)
        row["tvd_uniform"] = 0.5 * sum(abs(exact.get(k, 0.0) - c_u.get(k, 0) / len(burned_u)) for k in keys_u)
        row["tvd_mixture"] = 0.5 * sum(abs(exact.get(k, 0.0) - c_m.get(k, 0) / len(burned_m)) for k in keys_m)
        line += f"  |  TVD -- uniform: {row['tvd_uniform']:.3f}  mixture: {row['tvd_mixture']:.3f}"

    print(line)
    results.append(row)

print(f"\n=== aggregate over {len(results)} texts "
      f"(budget={BUDGET}, D={D}, eps={EPSILON}, alpha={ALPHA}) ===")
mean_u = sum(r["rate_uniform"] for r in results) / len(results)
mean_m = sum(r["rate_mixture"] for r in results) / len(results)
wins = sum(1 for r in results if r["rate_mixture"] > r["rate_uniform"])
print(f"mean target-visit rate -- uniform: {mean_u:.1%}  mixture: {mean_m:.1%}")
print(f"mixture beat uniform on {wins}/{len(results)} texts "
      f"(tied on {sum(1 for r in results if r['rate_mixture'] == r['rate_uniform'])})")

if EXACT_TVD:
    mean_tvd_u = sum(r["tvd_uniform"] for r in results) / len(results)
    mean_tvd_m = sum(r["tvd_mixture"] for r in results) / len(results)
    print(f"mean TVD to exact posterior -- uniform: {mean_tvd_u:.3f}  mixture: {mean_tvd_m:.3f}")

# --- Figure: one point per text, uniform's rate vs. mixture's rate. Above
# the diagonal = mixture won on that text, below = uniform won -- shows both
# how often AND by how much, which the two means alone would hide. --------
FIGURE = "results/figures/15_target_rate_comparison.png"
os.makedirs(os.path.dirname(FIGURE), exist_ok=True)

lo = 0.0
hi = max(max(r["rate_uniform"], r["rate_mixture"]) for r in results) * 1.15
colors = ["#1f5c8a" if r["rate_mixture"] > r["rate_uniform"] else "#c0783c"
          for r in results]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([lo, hi], [lo, hi], color="#93a4b8", linestyle="--", linewidth=1,
        label="equal performance")
ax.scatter([r["rate_uniform"] for r in results],
           [r["rate_mixture"] for r in results],
           c=colors, s=32, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
ax.set_xlabel("uniform proposer: target-visit rate")
ax.set_ylabel("mixture proposer: target-visit rate")
ax.set_title(f"Which proposer finds the right idiom more often? -- "
             f"{len(results)} texts, budget={BUDGET}")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.text(0.98, 0.02,
        f"mixture wins {wins}/{len(results)} texts\n"
        f"mean -- uniform: {mean_u:.1%}, mixture: {mean_m:.1%}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#4a5568")
fig.tight_layout()
fig.savefig(FIGURE, dpi=150)
print(f"\nfigure saved: {FIGURE}")
