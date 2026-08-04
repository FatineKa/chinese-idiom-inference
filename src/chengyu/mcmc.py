"""mcmc.py — Metropolis-Hastings loop. uniform_proposer is pure Python
   (no PyTorch/numpy dependency). text_proposer, below, depends on numpy
   and, via representation.py, on PyTorch — but only one Qwen forward pass
   at construction time, independent of the number of idioms."""

import math
import random
import time

import numpy as np

from chengyu.prior import log_prior
from chengyu.representation import text_state_by_layer
from chengyu.scoring import score_summary


def log_weight(text, idiom):
    return score_summary(text, idiom) + log_prior(idiom)   # score_summary = PyTorch, internally


def metropolis_hastings(text, initial_state, proposer, n_steps, seed=0, progress_every=None):
    """progress_every: if set, prints a status line every that many steps.
    Off by default, since this function is also called from contexts
    (e.g. a future API endpoint) where console output isn't wanted."""
    rng     = random.Random(seed)
    current = initial_state
    lw      = log_weight(text, current)
    trace   = []
    start   = time.time()
    for step in range(n_steps):
        candidate, log_hastings = proposer(current, rng)
        lw2 = log_weight(text, candidate)                    # 1 Qwen call (PyTorch inside)
        if math.log(rng.random()) < (lw2 - lw) + log_hastings:
            current, lw = candidate, lw2
        trace.append(current)
        if progress_every and (step + 1) % progress_every == 0:
            print(f"  step {step + 1}/{n_steps}  ({time.time() - start:.0f}s elapsed)")
    return trace


def uniform_proposer(idioms):
    n = len(idioms)
    def propose(current, rng):
        while True:
            candidate = idioms[rng.randrange(n)]
            if candidate != current:
                return candidate, 0.0
    return propose


def text_proposer(idioms: list, text: str, static_embeddings: np.ndarray,
                   layer: int, temperature: float = 1.0):
    """Informed proposer, constant cost: a single Qwen forward pass (on the
    text alone, here at construction time), compared against the idioms'
    static embeddings precomputed once (geometry.embeddings). `layer` must
    come from 09_classification_delta.py — do not pick one at random."""
    v = text_state_by_layer(text)[layer].cpu().numpy()
    norms = np.linalg.norm(static_embeddings, axis=1) * np.linalg.norm(v)
    weights = np.exp((static_embeddings @ v / norms) / temperature)
    weights /= weights.sum()
    index = {idiom: k for k, idiom in enumerate(idioms)}

    print(f"informed proposer built: layer {layer}, T={temperature}, "
          f"{len(idioms)} idioms, weight min={weights.min():.2e} "
          f"max={weights.max():.2e} (most favored idiom: {idioms[weights.argmax()]})")

    def propose(current, rng):
        idx = rng.choices(range(len(idioms)), weights=weights.tolist())[0]
        candidate = idioms[idx]
        log_hastings = math.log(weights[index[current]]) - math.log(weights[idx])
        return candidate, log_hastings
    return propose
