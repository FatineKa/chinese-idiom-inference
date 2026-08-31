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
import time
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
from chengyu.scoring import yes_no_judgment

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


def checkpoint_rate(indicator, checkpoints):
    """Running mean of a per-step 0/1 indicator, from step 1 through each
    checkpoint -- same shape as tvd_at_checkpoints, reused for acceptance
    rate and movement rate."""
    return [sum(indicator[:c]) / c for c in checkpoints]


def movement_indicator(trace, x0):
    """Per-step: did the state actually change? Differs from `accepted`
    for an independence sampler, which can accept a proposal that re-picks
    the current state (a self-loop) -- accepted without movement."""
    prev = [x0] + trace[:-1]
    return [cur != p for cur, p in zip(trace, prev)]


def first_hit_checkpoint(trace, key, checkpoints):
    """Per checkpoint: has `key` appeared anywhere in trace[:c] yet? A
    first-passage indicator (monotonically non-decreasing) -- an easier
    event than becoming the mode, so it separates 'never reached it' from
    'reached it but didn't stay'."""
    first_seen = next((m + 1 for m, x in enumerate(trace) if x == key), None)
    return [(first_seen is not None and first_seen <= c) for c in checkpoints]


run_rows = []                                  # one row per (text, beta, seed)
per_text_curves = {beta: [] for beta in BETAS}  # beta -> [mean-over-seeds curve, per text]
per_text_curves_iid = []                        # i.i.d.-from-pi_t reference, per text

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

    t0 = time.time()
    raw_h = text_scores(text, idioms)
    t_h = time.time() - t0
    h = {i: raw_h[i] + log_prior(i) for i in idioms}
    exact = exact_posterior(h)
    ranked = sorted(h, key=h.get, reverse=True)
    target_rank = ranked.index(target) + 1
    map_idiom = ranked[0]
    print(f"target posterior probability: {exact[target]:.4f}  (rank {target_rank}/{len(idioms)})  "
          f"MAP idiom: {map_idiom} (p={exact[map_idiom]:.4f})")

    # i.i.d.-from-pi_t reference: the TVD a PERFECT sampler would still show
    # at this many samples, purely from finite-sample noise -- no MCMC, no
    # extra Qwen calls, exact-posterior weights already computed above
    p_exact = np.array([exact[i] for i in idioms])
    iid_curves = []
    for r in range(N_SEEDS):
        iid_rng = np.random.default_rng(200_000 + SEED + text_id * 1000 + r)
        draws = iid_rng.choice(idioms, size=N_STEPS, p=p_exact).tolist()
        iid_curves.append(tvd_at_checkpoints(draws, exact, checkpoints))
    iid_curve = np.mean(iid_curves, axis=0)
    per_text_curves_iid.append(iid_curve)
    print(f"  {'iid ref':>10}: final TVD={iid_curve[-1]:.4f}  (finite-sample noise alone, "
          f"no MCMC)")

    h_cache = {(text, i): h[i] for i in idioms}   # every MCMC step becomes a lookup

    t0 = time.time()
    delta_scores = raw_delta_scores(idioms, text, layer=LAYER)
    t_delta = time.time() - t0
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

    t_mcmc = 0.0
    for beta in BETAS:
        name = "uniform" if beta == 0 else f"beta={beta}"
        proposer = (uniform_proposer(idioms) if beta == 0
                    else delta_proposer_from_scores(idioms, standardized_scores, beta))
        t0 = time.time()
        seed_curves, seed_visit, seed_mode_t, seed_mode_m, seed_judge = [], [], [], [], []
        seed_last_t, seed_last_m, seed_judge_last, seed_mode_eq_last = [], [], [], []
        seed_accept, seed_move, seed_maphit = [], [], []
        for r in range(N_SEEDS):
            trace, accepted = metropolis_hastings(text, x0_by_seed[r], proposer, N_STEPS,
                                                    seed=SEED + r, cache=dict(h_cache))
            curve = tvd_at_checkpoints(trace, exact, checkpoints)
            visit = trace.count(target) / len(trace)
            mode_t = tie_aware_match(trace, target)
            mode_m = tie_aware_match(trace, map_idiom)

            # acceptance vs. movement: differ exactly when an accepted
            # proposal re-picks the current state (possible for the
            # independence samplers, negligible in practice at |I|=31114
            # but tracked directly rather than assumed); MAP-hit is a
            # first-passage indicator, an easier event than mode-MAP
            accept_rate = checkpoint_rate(accepted, checkpoints)[-1]
            move_rate = checkpoint_rate(movement_indicator(trace, x0_by_seed[r]), checkpoints)[-1]
            map_hit = first_hit_checkpoint(trace, map_idiom, checkpoints)[-1]
            seed_accept.append(accept_rate)
            seed_move.append(move_rate)
            seed_maphit.append(map_hit)
            mode_idiom = Counter(trace).most_common(1)[0][0]   # chain's own answer,
                                # one concrete idiom to ask the judge about
                                # (ties broken arbitrarily -- fine here, we
                                # just need a single candidate)
            judged_yes, log_p_yes, log_p_no = yes_no_judgment(text, mode_idiom)

            # last state: reported alongside the mode, not instead of it -- a
            # single draw from the chain, not a "converged answer" (a
            # well-mixed chain keeps sampling, it doesn't settle), but worth
            # having as a point of comparison
            last_state = trace[-1]
            last_t = float(last_state == target)
            last_m = float(last_state == map_idiom)
            mode_eq_last = (last_state == mode_idiom)
            if mode_eq_last:
                judged_yes_last, log_p_yes_last, log_p_no_last = judged_yes, log_p_yes, log_p_no
            else:
                judged_yes_last, log_p_yes_last, log_p_no_last = yes_no_judgment(text, last_state)

            seed_curves.append(curve)
            seed_visit.append(visit)
            seed_mode_t.append(mode_t)
            seed_mode_m.append(mode_m)
            seed_judge.append(judged_yes)
            seed_last_t.append(last_t)
            seed_last_m.append(last_m)
            seed_judge_last.append(judged_yes_last)
            seed_mode_eq_last.append(mode_eq_last)
            run_rows.append({"text_id": text_id, "target": target, "target_rank": target_rank,
                              "map_idiom": map_idiom, "beta": beta, "seed": r,
                              "final_tvd": curve[-1], "target_visit_rate": visit,
                              "mode_target_match": mode_t, "mode_map_match": mode_m,
                              "mode_idiom": mode_idiom, "judge_approves_mode": judged_yes,
                              "judge_margin": log_p_yes - log_p_no,
                              "last_state": last_state, "last_target_match": last_t,
                              "last_map_match": last_m, "judge_approves_last": judged_yes_last,
                              "last_judge_margin": log_p_yes_last - log_p_no_last,
                              "mode_equals_last": mode_eq_last,
                              "acceptance_rate": accept_rate, "movement_rate": move_rate,
                              "map_hit_ever": map_hit})
        t_mcmc += time.time() - t0
        text_curve = np.mean(seed_curves, axis=0)
        per_text_curves[beta].append(text_curve)
        print(f"  {name:>10}: final TVD={text_curve[-1]:.4f}  "
              f"visit_rate={np.mean(seed_visit):.1%}  "
              f"mode-target={np.mean(seed_mode_t):.1%}  mode-MAP={np.mean(seed_mode_m):.1%}  "
              f"judge-approves(mode)={np.mean(seed_judge):.1%}")
        print(f"  {'':>10}  last-target={np.mean(seed_last_t):.1%}  "
              f"last-MAP={np.mean(seed_last_m):.1%}  "
              f"judge-approves(last)={np.mean(seed_judge_last):.1%}  "
              f"mode==last={np.mean(seed_mode_eq_last):.1%}")
        print(f"  {'':>10}  acceptance={np.mean(seed_accept):.1%}  "
              f"movement={np.mean(seed_move):.1%}  "
              f"MAP-hit-ever={np.mean(seed_maphit):.1%}")

    print(f"  timing: h_t sweep={t_h:.1f}s  delta sweep={t_delta:.1f}s  "
          f"all-beta MCMC loops={t_mcmc:.1f}s")

results = pd.DataFrame(run_rows)
os.makedirs("results/outputs", exist_ok=True)
results.to_csv("results/outputs/20_results.csv", index=False)
print(f"\nsaved {len(results)} rows to results/outputs/20_results.csv")

# cross-text aggregation: seeds already averaged into per_text_curves above;
# here, average each beta's per-text values across texts
print(f"\n=== aggregated over {text_id} text(s) ===")
metrics = ["final_tvd", "target_visit_rate", "mode_target_match", "mode_map_match",
           "judge_approves_mode", "last_target_match", "last_map_match",
           "judge_approves_last", "mode_equals_last",
           "acceptance_rate", "movement_rate", "map_hit_ever"]
per_text_by_beta = {beta: results[results["beta"] == beta].groupby("text_id")[metrics].mean()
                     for beta in BETAS}
for beta in BETAS:
    name = "uniform" if beta == 0 else f"beta={beta}"
    agg = per_text_by_beta[beta].mean()
    print(f"  {name:>10}: TVD={agg['final_tvd']:.4f}  visit_rate={agg['target_visit_rate']:.1%}  "
          f"mode-target={agg['mode_target_match']:.1%}  mode-MAP={agg['mode_map_match']:.1%}  "
          f"judge-approves(mode)={agg['judge_approves_mode']:.1%}")
    print(f"  {'':>10}  last-target={agg['last_target_match']:.1%}  "
          f"last-MAP={agg['last_map_match']:.1%}  "
          f"judge-approves(last)={agg['judge_approves_last']:.1%}  "
          f"mode==last={agg['mode_equals_last']:.1%}")
    print(f"  {'':>10}  acceptance={agg['acceptance_rate']:.1%}  "
          f"movement={agg['movement_rate']:.1%}  "
          f"MAP-hit-ever={agg['map_hit_ever']:.1%}")

iid_final = np.mean([c[-1] for c in per_text_curves_iid])
print(f"  {'iid ref':>10}: TVD={iid_final:.4f}  (finite-sample noise alone, no MCMC)")

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
iid_curves = np.array(per_text_curves_iid)
iid_mean = iid_curves.mean(axis=0)
ax.plot(checkpoints, iid_mean, marker="o", markersize=3, linewidth=2, linestyle="--",
        color="black", label="i.i.d. reference")
if len(iid_curves) > 1:
    iid_std = iid_curves.std(axis=0)
    ax.fill_between(checkpoints, iid_mean - iid_std, iid_mean + iid_std, alpha=0.15, color="black")
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
