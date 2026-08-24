"""20_delta_proposal_comparison.py -- does the Delta_23^E-informed
independence proposal q_beta approximate the true posterior faster (in
step count) than the uniform proposal, once both are pure lookups
(Delta_23^E and h_t precomputed once per text, shared across the whole
beta/seed sweep)?

TVD tracks the running (from step 1, no burn-in) time-average occupation
measure against the exact posterior -- an ergodic-average statistic
(Levin & Peres), not marginal-law mixing-time convergence.

Aggregation order: average seeds within each text first, then average
across texts -- so no text counts for more just because it ran more
seeds. Every (text, beta, seed) result is saved to
results/outputs/20_results.csv for reproducibility.

Cost per text: O(|I|) likelihood calls (exact posterior + MCMC target,
shared via one cache) + O(|I|) representation calls (Delta_23^E) --
batched, but still expensive; start with 1-3 texts."""
import os
from collections import Counter

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
    """TVD at each checkpoint, counted from step 1 -- convergence speed
    from a cold start is the claim under test, not noise to burn away."""
    out = []
    for c in checkpoints:
        counts = Counter(trace[:c])
        out.append(0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / c)
                              for k in set(exact) | set(counts)))
    return out


def tie_aware_match(trace, key):
    """Fractional credit if `key` is among the trace's (possibly tied)
    most-visited idioms -- Counter.most_common(1) alone breaks ties
    arbitrarily by encounter order."""
    counts = Counter(trace)
    best = max(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    return (1.0 / len(winners)) if key in winners else 0.0


run_rows = []                                  # one row per (text, beta, seed)
per_text_curves = {beta: [] for beta in BETAS}  # beta -> [mean-over-seeds curve, per text]

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

    raw_h = text_scores(text, idioms)
    h = {i: raw_h[i] + log_prior(i) for i in idioms}
    exact = exact_posterior(h)
    ranked = sorted(h, key=h.get, reverse=True)
    target_rank = ranked.index(target) + 1
    map_idiom = ranked[0]
    print(f"target posterior probability: {exact[target]:.4f}  (rank {target_rank}/{len(idioms)})  "
          f"MAP idiom: {map_idiom} (p={exact[map_idiom]:.4f})")

    h_cache = {(text, i): h[i] for i in idioms}   # every MCMC step becomes a lookup

    delta_scores = raw_delta_scores(idioms, text, layer=LAYER)
    by_length = {}
    for i in idioms:
        by_length.setdefault(len(i), []).append(i)
    # standardized WITHIN each length class -- dictionary is 95.4% length-4,
    # a global mean/std would just be that population's statistics
    standardized_scores = {}
    for n, group in by_length.items():
        vals = np.array([delta_scores[i] for i in group])
        mu_n, sigma_n = vals.mean(), max(vals.std(), 1e-8)
        for i in group:
            standardized_scores[i] = (delta_scores[i] - mu_n) / sigma_n

    # one starting idiom PER SEED, reused across every beta: pairs the
    # beta comparison (same start for beta=0 and beta=5 at seed r) while
    # still checking sensitivity to the starting point, across seeds
    x0_rng = np.random.default_rng(SEED + text_id)
    x0_by_seed = [idioms[k] for k in x0_rng.integers(len(idioms), size=N_SEEDS)]

    for beta in BETAS:
        name = "uniform" if beta == 0 else f"beta={beta}"
        proposer = (uniform_proposer(idioms) if beta == 0
                    else delta_proposer_from_scores(idioms, standardized_scores, beta))
        seed_curves, seed_visit, seed_mode_t, seed_mode_m = [], [], [], []
        for r in range(N_SEEDS):
            trace, _ = metropolis_hastings(text, x0_by_seed[r], proposer, N_STEPS,
                                            seed=SEED + r, cache=dict(h_cache))
            curve = tvd_at_checkpoints(trace, exact, checkpoints)
            visit = trace.count(target) / len(trace)
            mode_t = tie_aware_match(trace, target)
            mode_m = tie_aware_match(trace, map_idiom)
            seed_curves.append(curve)
            seed_visit.append(visit)
            seed_mode_t.append(mode_t)
            seed_mode_m.append(mode_m)
            run_rows.append({"text_id": text_id, "target": target, "target_rank": target_rank,
                              "map_idiom": map_idiom, "beta": beta, "seed": r,
                              "final_tvd": curve[-1], "target_visit_rate": visit,
                              "mode_target_match": mode_t, "mode_map_match": mode_m})
        text_curve = np.mean(seed_curves, axis=0)
        per_text_curves[beta].append(text_curve)
        print(f"  {name:>10}: final TVD={text_curve[-1]:.4f}  "
              f"visit_rate={np.mean(seed_visit):.1%}  "
              f"mode-target={np.mean(seed_mode_t):.1%}  mode-MAP={np.mean(seed_mode_m):.1%}")

results = pd.DataFrame(run_rows)
os.makedirs("results/outputs", exist_ok=True)
results.to_csv("results/outputs/20_results.csv", index=False)
print(f"\nsaved {len(results)} rows to results/outputs/20_results.csv")

# cross-text aggregation: seeds already averaged into per_text_curves above;
# here, average each beta's per-text values across texts
print(f"\n=== aggregated over {text_id} text(s) ===")
metrics = ["final_tvd", "target_visit_rate", "mode_target_match", "mode_map_match"]
per_text_by_beta = {beta: results[results["beta"] == beta].groupby("text_id")[metrics].mean()
                     for beta in BETAS}
for beta in BETAS:
    name = "uniform" if beta == 0 else f"beta={beta}"
    agg = per_text_by_beta[beta].mean()
    print(f"  {name:>10}: TVD={agg['final_tvd']:.4f}  visit_rate={agg['target_visit_rate']:.1%}  "
          f"mode-target={agg['mode_target_match']:.1%}  mode-MAP={agg['mode_map_match']:.1%}")

# paired difference per text: TVD(beta) - TVD(uniform) -- % of texts where
# the informed proposal actually beat uniform, not just the pooled average
print("\npercentage of texts where informed TVD < uniform TVD:")
uniform_tvd = per_text_by_beta[0]["final_tvd"]
for beta in BETAS:
    if beta == 0:
        continue
    diff = per_text_by_beta[beta]["final_tvd"] - uniform_tvd
    print(f"  beta={beta}: {(diff < 0).mean():.1%}  (mean diff={diff.mean():+.4f}, n={len(diff)} texts)")

fig, ax = plt.subplots(figsize=(7, 4.5))
for beta in BETAS:
    name = "uniform" if beta == 0 else f"beta={beta}"
    curves = np.array(per_text_curves[beta])   # (n_texts, len(checkpoints))
    mean_curve = curves.mean(axis=0)
    ax.plot(checkpoints, mean_curve, marker="o", markersize=3, linewidth=2, label=name)
    if len(curves) > 1:
        std_curve = curves.std(axis=0)
        ax.fill_between(checkpoints, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
ax.set_xlabel("MCMC step")
ax.set_ylabel(f"TVD to exact posterior (mean over {text_id} text(s), {N_SEEDS} seeds each)")
ax.set_title("Posterior-approximation efficiency: uniform vs. representation-informed")
ax.set_ylim(bottom=0.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
figure = "results/figures/20_tvd_vs_steps.png"
os.makedirs(os.path.dirname(figure), exist_ok=True)
fig.savefig(figure, dpi=150)
print(f"figure saved: {figure}")
