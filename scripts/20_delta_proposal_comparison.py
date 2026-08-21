"""20_delta_proposal_comparison.py -- does the representation-informed
independence sampler q_beta (chapter sections sec:delta_proposal through
sec:delta_proposal_experiment) reach the true posterior in FEWER STEPS than
uniform, once its one-time setup cost is treated as given rather than
something to economize on?

Uniform and every beta in the sweep run in the SAME script, on the SAME
text, sharing the same exact posterior and the same Delta_23^E scores --
same reasoning as script 15 comparing uniform and mixture together rather
than as two separate runs.

Comparison axis is STEP COUNT, not total model calls (unlike script 15's
budget-matched comparison): building q_beta already costs as much as
computing the exact posterior directly (see sec:delta_why_help), so it
cannot be framed as a cheaper alternative to enumeration. What CAN be
tested fairly is mixing speed -- once Delta_23^E(i,t) is known for every
idiom (paid once per text, shared across the whole beta sweep) and h_t(i)
values get cached as they're visited, every further MCMC step is cheap for
every proposer alike, so comparing them at equal step count is the fair
axis here, not equal cost.

Cost per text: ~31,113 calls for Delta_23^E (raw_delta_scores, shared by
the whole beta sweep) + ~31,113 calls for the exact posterior (text_scores,
shared by every proposer's TVD). Expensive; start with 1-3 texts, not 50."""
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
BETAS = [float(b) for b in os.environ.get("CHENGYU_BETAS", "0,1,5,20").split(",")]
N_STEPS = int(os.environ.get("CHENGYU_N_STEPS", "3000"))
N_CHECKPOINTS = int(os.environ.get("CHENGYU_N_CHECKPOINTS", "15"))
N_TEXTS = int(os.environ.get("CHENGYU_N_TEXTS", "1"))
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
    """TVD at each checkpoint, computed from PREFIXES of one already-run
    trace -- avoids rerunning the chain once per checkpoint. Burn-in (10%)
    is taken relative to each checkpoint's own prefix length, not the full
    trace, so an early checkpoint isn't dominated by steps that would count
    as burn-in at the full length."""
    from collections import Counter
    out = []
    for c in checkpoints:
        prefix = trace[:c]
        burn = max(1, len(prefix) // 10)
        burned = prefix[burn:]
        counts = Counter(burned)
        tvd = 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / len(burned))
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

    print(f"computing Delta_{LAYER}^E for every idiom (shared across the beta sweep)...")
    delta_scores = raw_delta_scores(idioms, text, layer=LAYER)

    curves = {}
    for beta in BETAS:
        name = "uniform" if beta == 0 else f"beta={beta}"
        proposer = (uniform_proposer(idioms) if beta == 0
                    else delta_proposer_from_scores(idioms, delta_scores, beta))
        trace, accepted = metropolis_hastings(text, target, proposer, N_STEPS, seed=SEED)
        curve = tvd_at_checkpoints(trace, exact, checkpoints)
        acc_rate = sum(accepted) / len(accepted)
        curves[name] = curve
        print(f"  {name:>10}: final TVD={curve[-1]:.4f}  acceptance={acc_rate:.1%}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in curves.items():
        ax.plot(checkpoints, curve, marker="o", markersize=3, linewidth=2, label=name)
    ax.set_xlabel("MCMC step")
    ax.set_ylabel("TVD to exact posterior")
    ax.set_title(f"Mixing speed: uniform vs. representation-informed proposal -- "
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
