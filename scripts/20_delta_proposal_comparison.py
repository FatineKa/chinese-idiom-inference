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

Cost per text: |I|=31,114 candidate-level likelihood evaluations
(text_scores, for the exact posterior AND reused as the MCMC target via a
precomputed cache) + |I|=31,114 candidate-level representation evaluations
(raw_delta_scores, for Delta_23^E) -- batched, so not 31,114 separate
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


def tvd_at_checkpoints(trace, exact, checkpoints):
    """TVD at each checkpoint, counting from step 1 -- no burn-in discarded.
    The claim being tested is "does the informed proposal get close to the
    truth in fewer steps, starting cold", and the early steps (climbing out
    of a bad, randomly-chosen starting idiom) ARE that claim, not noise to
    exclude before measuring it."""
    from collections import Counter
    out = []
    for c in checkpoints:
        samples = trace[:c]
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
    print(f"Delta_{LAYER}^E distribution: min={raw_deltas.min():.4f} "
          f"p10={np.quantile(raw_deltas, 0.1):.4f} median={np.median(raw_deltas):.4f} "
          f"p90={np.quantile(raw_deltas, 0.9):.4f} max={raw_deltas.max():.4f} "
          f"std={raw_deltas.std():.4f}")
    # standardized WITHIN each idiom length class, not globally: the
    # representation study that validated Delta_23^E only ever compared
    # idioms of the SAME length (length-matched distractors), and the
    # dictionary is 95.4% length-4 idioms -- a single global mean/std would
    # be almost entirely the length-4 population's statistics, silently
    # applied to every other length too. This keeps beta comparable across
    # texts (still an affine rescaling, so ranking within a length class is
    # unchanged) without importing a length-based bias the validation never
    # tested for.
    by_length = {}
    for i in idioms:
        by_length.setdefault(len(i), []).append(i)
    standardized_scores = {}
    for n, group in by_length.items():
        vals = np.array([delta_scores[i] for i in group])
        mu_n, sigma_n = vals.mean(), max(vals.std(), 1e-8)
        for i in group:
            standardized_scores[i] = (delta_scores[i] - mu_n) / sigma_n

    # one shared starting state, drawn independently of the target -- NOT
    # the known-correct idiom, so early-step behavior isn't inflated by
    # starting the chain already at the answer
    x0_rng = np.random.default_rng(SEED)
    x0 = idioms[x0_rng.integers(len(idioms))]
    print(f"shared starting state (independent of target): {x0}")

    curves = {}          # name -> (mean TVD curve, std TVD curve)
    visit_rates = {}     # name -> (mean target-visit rate, std) -- how much
                          # of its time the chain spends on the real answer
    top1_rates = {}       # name -> fraction of seeds whose MOST-VISITED
                          # idiom (the chain's single "final answer",
                          # analogous to the ranking system's top-1) matches
                          # the target -- NOT the last state (see below)
    for beta in BETAS:
        name = "uniform" if beta == 0 else f"beta={beta}"
        seed_curves = []
        seed_visit_rates = []
        seed_top1_matches = []
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
            seed_curves.append(tvd_at_checkpoints(trace, exact, checkpoints))
            seed_visit_rates.append(trace.count(target) / len(trace))   # counted
                        # from step 1, same as the TVD curve -- no burn-in
                        # discarded, for the same reason
            # "top-1" for one chain = its single MOST-VISITED idiom, not its
            # LAST state: a well-mixed chain keeps wandering according to
            # the whole distribution, it doesn't settle down and stay put,
            # so the very last state is close to a one-sample coin flip
            # (noisy, easy to get unlucky) -- the mode is the chain's
            # actual "best single guess" if forced to name one.
            from collections import Counter
            mode_idiom, _ = Counter(trace).most_common(1)[0]
            seed_top1_matches.append(mode_idiom == target)
        seed_curves = np.array(seed_curves)   # (N_SEEDS, len(checkpoints))
        mean_curve = seed_curves.mean(axis=0)
        curves[name] = (mean_curve, seed_curves.std(axis=0))
        seed_visit_rates = np.array(seed_visit_rates)
        visit_rates[name] = (seed_visit_rates.mean(), seed_visit_rates.std())
        top1_rates[name] = np.mean(seed_top1_matches)
        acc_rate = sum(accepted) / len(accepted)   # last seed's, as a diagnostic only
        print(f"  {name:>10}: final TVD={mean_curve[-1]:.4f} "
              f"(+/- {seed_curves.std(axis=0)[-1]:.4f} across {N_SEEDS} seeds)  "
              f"target-visit rate={seed_visit_rates.mean():.1%} "
              f"(+/- {seed_visit_rates.std():.1%})  "
              f"top-1 (mode) match={top1_rates[name]:.1%}  "
              f"last-seed acceptance={acc_rate:.1%}")

    print(f"\nhow often each method's chain was actually sitting on the "
          f"corpus reference idiom ({target}):")
    for name, (mean_rate, std_rate) in visit_rates.items():
        print(f"  {name:>10}: {mean_rate:.1%} (+/- {std_rate:.1%})")

    print(f"\nhow often each method's single best guess (most-visited idiom) "
          f"IS the corpus reference idiom ({target}), across {N_SEEDS} seeds:")
    for name, rate in top1_rates.items():
        print(f"  {name:>10}: {rate:.1%}")

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
