"""20_delta_proposal_comparison.py -- does the representation-informed
independence sampler q_beta (chapter sections sec:delta_proposal through
sec:delta_proposal_experiment) produce a more accurate empirical
approximation of the true posterior after fewer MCMC transitions than
uniform, once the required quantities have been precomputed?

Uniform and every beta in the sweep run in the SAME script, on the SAME
text, sharing the same exact posterior and the same Delta_23^E scores --
same reasoning as script 15 comparing uniform and mixture together rather
than as two separate runs.

Comparison axis is STEP COUNT, not total model calls (unlike script 15's
budget-matched comparison): building q_beta already costs as much as
computing the exact posterior directly (see sec:delta_why_help), so it
cannot be framed as a cheaper alternative to enumeration. What CAN be
tested fairly is posterior-approximation efficiency of the transition
kernel itself -- once Delta_23^E(i,t) is known for every idiom (paid once
per text, shared across the whole beta sweep) AND h_t(i) is known for
every idiom too (also computed once, then passed into every MCMC run as a
precomputed cache -- see USE_PRECOMPUTED_H below), every step is a pure
lookup for every proposer alike, so comparing them at equal step count is
the fair axis here, not equal cost.

The statistic plotted, d_TV(empirical occupation measure, exact posterior),
is the time-average of one chain converging by the ergodic theorem -- NOT
the marginal law of X_m converging in the usual mixing-time sense (Levin &
Peres). Reported as such; several seeds are averaged (CHENGYU_N_SEEDS) to
reduce the noise of relying on one trajectory, cheap to do since every step
is a lookup once the two O(|I|) precomputations are done.

Cost per text: |I|=31,113 candidate-level likelihood evaluations
(text_scores, for the exact posterior AND reused as the MCMC target via a
precomputed cache) + |I|=31,113 candidate-level representation evaluations
(raw_delta_scores, for Delta_23^E) -- batched, so not 31,113 separate
network invocations, but still O(|I|) work either way. Expensive; start
with 1-3 texts, not 50."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chengyu.argmax import exact_posterior, text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.mcmc import (
    delta_proposer_from_scores, metropolis_hastings, raw_delta_scores, uniform_proposer,
)
from chengyu.prior import log_prior

LAYER = int(os.environ.get("CHENGYU_LAYER", "23"))
BETAS = [float(b) for b in os.environ.get("CHENGYU_BETAS", "0,0.5,1,2,5").split(",")]
N_STEPS = int(os.environ.get("CHENGYU_N_STEPS", "3000"))
N_CHECKPOINTS = int(os.environ.get("CHENGYU_N_CHECKPOINTS", "15"))
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "1"))
N_SEEDS = int(os.environ.get("CHENGYU_N_SEEDS", "5"))
BURN_IN = int(os.environ.get("CHENGYU_BURN_IN", str(N_STEPS // 10)))   # FIXED
                        # across checkpoints, not 10% of each checkpoint's
                        # own length -- otherwise a sample counted at one
                        # checkpoint could be discarded as burn-in at
                        # another, and the "kept" window wouldn't simply
                        # grow as more steps are taken
SEED = 0

dictionary, lengths = load_dictionary()
idioms = sorted(dictionary)

df = (
    pd.read_csv("data/raw/cip/in_domain/test.in.csv")
    .drop_duplicates(subset=["src", "dst"])
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

checkpoints = sorted(set(
    int(round(N_STEPS * frac)) for frac in np.linspace(1 / N_CHECKPOINTS, 1.0, N_CHECKPOINTS)
))
checkpoints = [c for c in checkpoints if c > BURN_IN]


def tvd_at_checkpoints(trace, exact, checkpoints, burn_in):
    """TVD at each checkpoint, from a FIXED burn-in: the window
    trace[burn_in:c] only grows as c grows, unlike a burn-in taken relative
    to each checkpoint's own length."""
    from collections import Counter
    out = []
    for c in checkpoints:
        samples = trace[burn_in:c]
        counts = Counter(samples)
        tvd = 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / len(samples))
                         for k in set(exact) | set(counts))
        out.append(tvd)
    return out


text_id = 0
for src, dst in zip(df["src"], df["dst"]):
    if text_id >= N_TEXTS:
        break
    target = find_idiom(src, dst, dictionary, lengths)
    if not target:
        continue
    text = normalize(dst)
    text_id += 1
    print(f"\n=== text {text_id}/{N_TEXTS} -- target: {target} ===")

    print("computing exact posterior over the full dictionary...")
    raw_h = text_scores(text, idioms)
    h = {i: raw_h[i] + log_prior(i) for i in idioms}
    exact = exact_posterior(h)
    rank = sorted(h, key=h.get, reverse=True).index(target) + 1
    print(f"target posterior probability: {exact[target]:.4f}  (rank {rank}/{len(idioms)})")

    # USE_PRECOMPUTED_H: pre-populate metropolis_hastings' cache with every
    # idiom's h_t so every MCMC step is a lookup, not a fresh model call --
    # without this, "steps are free once setup is paid" is false in practice.
    h_cache = {(text, i): h[i] for i in idioms}

    print(f"computing Delta_{LAYER}^E for every idiom (shared across the beta sweep)...")
    delta_scores = raw_delta_scores(idioms, text, layer=LAYER)

    raw_deltas = np.array(list(delta_scores.values()))
    mu_delta, sigma_delta = raw_deltas.mean(), max(raw_deltas.std(), 1e-8)
    print(f"Delta_{LAYER}^E distribution: min={raw_deltas.min():.4f} "
          f"p10={np.quantile(raw_deltas, 0.1):.4f} median={np.median(raw_deltas):.4f} "
          f"p90={np.quantile(raw_deltas, 0.9):.4f} max={raw_deltas.max():.4f} "
          f"std={raw_deltas.std():.4f}")
    # standardized so beta is comparable across texts/layers -- an affine
    # rescaling of the SAME score, so it does not change which candidate
    # gets the most weight at a given relative strength, only makes beta's
    # scale interpretable (unlike raw Delta, whose scale varies by text)
    standardized_scores = {i: (d - mu_delta) / sigma_delta for i, d in delta_scores.items()}

    # one shared starting state, drawn independently of the target -- NOT
    # the known-correct idiom, so early-step behavior isn't inflated by
    # starting the chain already at the answer
    x0_rng = np.random.default_rng(SEED)
    x0 = idioms[x0_rng.integers(len(idioms))]
    print(f"shared starting state (independent of target): {x0}")

    curves = {}   # name -> list of per-seed TVD curves
    for beta in BETAS:
        name = "uniform" if beta == 0 else f"beta={beta}"
        seed_curves = []
        for r in range(N_SEEDS):
            proposer = (uniform_proposer(idioms) if beta == 0
                        else delta_proposer_from_scores(idioms, standardized_scores, beta))
            run_seed = SEED + r
            trace, accepted = metropolis_hastings(text, x0, proposer, N_STEPS,
                                                    seed=run_seed, cache=dict(h_cache))
                        # dict(h_cache): a fresh copy per run -- the cache is
                        # only ever read from here (every key already
                        # present), but copying avoids any risk of one run's
                        # writes leaking into another's
            seed_curves.append(tvd_at_checkpoints(trace, exact, checkpoints, BURN_IN))
        seed_curves = np.array(seed_curves)   # (N_SEEDS, len(checkpoints))
        mean_curve = seed_curves.mean(axis=0)
        curves[name] = (mean_curve, seed_curves.std(axis=0))
        acc_rate = sum(accepted) / len(accepted)   # last seed's, as a diagnostic only
        print(f"  {name:>10}: final TVD={mean_curve[-1]:.4f} "
              f"(+/- {seed_curves.std(axis=0)[-1]:.4f} across {N_SEEDS} seeds)  "
              f"last-seed acceptance={acc_rate:.1%}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (mean_curve, std_curve) in curves.items():
        ax.plot(checkpoints, mean_curve, marker="o", markersize=3, linewidth=2, label=name)
        ax.fill_between(checkpoints, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    ax.set_xlabel("MCMC step")
    ax.set_ylabel(f"TVD to exact posterior (mean +/- std, {N_SEEDS} seeds)")
    ax.set_title(f"Posterior-approximation efficiency: uniform vs. representation-informed -- "
                 f"layer {LAYER}, text {text_id}")
    ax.set_ylim(bottom=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure = f"results/figures/20_tvd_vs_steps_text{text_id}.png"
    os.makedirs(os.path.dirname(figure), exist_ok=True)
    fig.savefig(figure, dpi=150)
    print(f"  figure saved: {figure}")
