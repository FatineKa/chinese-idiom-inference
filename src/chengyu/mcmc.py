"""mcmc.py — Metropolis-Hastings loop. uniform_proposer is pure Python
   (no PyTorch/numpy dependency). text_proposer, below, depends on numpy
   and, via representation.py, on PyTorch — but only one Qwen forward pass
   at construction time, independent of the number of idioms."""

import itertools
import math
import random
import time

import numpy as np

from chengyu.geometry import static_embedding_stats
from chengyu.prior import log_prior
from chengyu.representation import (
    context_states_batch, euclidean_delta, load_text_state_stats, text_state_by_layer,
)
from chengyu.scoring import score_summary


def log_weight(text, idiom, cache=None):
    """h_t(idiom) = log p_theta(text|idiom) + log p(idiom). cache: optional
    {(text, idiom): value} dict, shared between metropolis_hastings' own
    acceptance-check calls and a proposer's internal scoring (e.g.
    mixture_proposer scoring a neighborhood) -- so the same idiom scored
    by both is only ever a real model call once."""
    if cache is not None and (text, idiom) in cache:
        return cache[(text, idiom)]
    value = score_summary(text, idiom) + log_prior(idiom)   # score_summary = PyTorch, internally
    if cache is not None:
        cache[(text, idiom)] = value
    return value


def edit_distance_1_graph(idioms: list) -> dict:
    """{idiom: set of neighbors} at edit distance 1 (substitution,
    insertion, or deletion of exactly one character), over the WHOLE
    dictionary (all lengths). Checked directly on this project's
    dictionary: ~47.7% of idioms are isolated at D=1, and this does not
    improve at D=2 or D=3 -- those idioms sit in their own disconnected
    component, not just "far" -- so the local-edit proposal alone cannot
    be irreducible; see mixture_proposer's length-aware global jump."""
    by_length = {}
    for i in idioms:
        by_length.setdefault(len(i), set()).add(i)

    adj = {i: set() for i in idioms}

    # substitution: same length, differ at exactly one position
    for n, group in by_length.items():
        group_list = list(group)
        for j in range(n):
            patterns = {}
            for idiom in group_list:
                patterns.setdefault(idiom[:j] + "\0" + idiom[j + 1:], []).append(idiom)
            for members in patterns.values():
                if len(members) > 1:
                    for a in members:
                        adj[a].update(m for m in members if m != a)

    # insertion/deletion: idiom with one character removed matches a
    # dictionary entry one character shorter
    for idiom in idioms:
        n = len(idiom)
        shorter_group = by_length.get(n - 1)
        if shorter_group:
            for i in range(n):
                shorter = idiom[:i] + idiom[i + 1:]
                if shorter in shorter_group:
                    adj[idiom].add(shorter)
                    adj[shorter].add(idiom)

    return adj


def _bfs_ball(adj: dict, start: str, radius: int) -> set:
    """Nodes reachable from `start` within `radius` hops of `adj` (excludes
    `start` itself)."""
    visited = {start}
    frontier = {start}
    for _ in range(radius):
        next_frontier = set()
        for node in frontier:
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    visited.discard(start)
    return visited


def mixture_proposer(idioms: list, text: str, D: int = 1, epsilon: float = 0.1,
                      alpha: float = 0.5, temperature_loc: float = 1.0, cache=None):
    """q_eps(y|x,t) = (1-eps) q_loc(y|x,t) + eps q_len(y).

    q_loc: softmax, temperature temperature_loc, of h_t(y) = log_weight(text, y)
    (the SAME target score used everywhere else -- no separate scoring
    scheme) over N_D(x), the idioms within edit distance D of x. This
    directly asks "among dictionary idioms structurally close to x, which
    have good posterior scores for text t", rather than generating a
    replacement character in isolation.

    q_len: a length-aware global jump, so the chain can reach idioms of
    ANY length (and any idiom at all, including the ~47.7% with no local
    neighbor) regardless of how skewed the dictionary's length
    distribution is: sample a length n with probability proportional to
    |I_n|^alpha (alpha=1 recovers the plain uniform-over-dictionary
    distribution; alpha=0 makes every length equally likely), then sample
    uniformly among idioms of that length.

    When N_D(x) is empty, the local branch has nothing to propose, so it
    proposes staying at x (a self-loop, matching the theory's Algorithm 2:
    "if N_D(x) = empty, set y <- x") -- the (1-eps) local weight is not
    silently redirected into q_len. q_eps(y|x,t) is still a valid,
    correctly normalised distribution, and still > 0 for every x, y
    (via the eps * q_len(y) term alone), so irreducibility holds
    regardless of the edit-distance graph's disconnected components."""
    adj = edit_distance_1_graph(idioms)
    balls = {x: _bfs_ball(adj, x, D) for x in idioms}

    by_length = {}
    for i in idioms:
        by_length.setdefault(len(i), []).append(i)
    lengths = list(by_length)
    raw = {n: len(by_length[n]) ** alpha for n in lengths}
    z_len = sum(raw.values())
    nu = {n: raw[n] / z_len for n in lengths}

    def q_len_prob(idiom):
        return nu[len(idiom)] / len(by_length[len(idiom)])

    def sample_q_len(rng):
        n = rng.choices(lengths, weights=[nu[l] for l in lengths])[0]
        return rng.choice(by_length[n])

    def q_loc_probs(x):
        """{y: probability} over N_D(x), or None if x has no local neighbor."""
        ball = balls[x]
        if not ball:
            return None
        scores = {y: log_weight(text, y, cache=cache) for y in ball}
        m = max(scores.values())
        weights = {y: math.exp((s - m) / temperature_loc) for y, s in scores.items()}
        z = sum(weights.values())
        return {y: w / z for y, w in weights.items()}

    def q_prob(target, source, source_loc_probs):
        """q_eps(target | source, t), given source's own q_loc_probs (or
        None if source has no local neighbor). When source has no local
        neighbor, the LOCAL branch proposes staying at source (Algorithm 2:
        "if N_D(x) = empty, set y <- x"), not a fallback to q_len -- so
        the (1-eps) local weight lands entirely on target == source in
        that case, not spread over q_len."""
        if source_loc_probs is None:
            self_loop = (1 - epsilon) if target == source else 0.0
            return self_loop + epsilon * q_len_prob(target)
        loc = source_loc_probs.get(target, 0.0)
        return (1 - epsilon) * loc + epsilon * q_len_prob(target)

    def propose(current, rng):
        loc_probs = q_loc_probs(current)   # needed either way: even a
                                            # y drawn from q_len can still
                                            # carry local probability mass
        if rng.random() < epsilon:
            y = sample_q_len(rng)
        elif loc_probs is None:
            y = current                    # local branch, no neighbor: self-loop
        else:
            candidates = list(loc_probs)
            y = rng.choices(candidates, weights=[loc_probs[c] for c in candidates])[0]

        y_loc_probs = q_loc_probs(y)
        fwd = q_prob(y, current, loc_probs)
        rev = q_prob(current, y, y_loc_probs)
        log_hastings = math.log(rev) - math.log(fwd)
        return y, log_hastings

    return propose


def metropolis_hastings(text, initial_state, proposer, n_steps, seed=0,
                         progress_every=None, cache=None):
    """progress_every: if set, prints a status line every that many steps.
    Off by default, since this function is also called from contexts
    (e.g. a future API endpoint) where console output isn't wanted.
    cache: optional {(text, idiom): h_t(idiom)} dict, shared with the
    proposer (e.g. mixture_proposer) if it also scores idioms, so a
    candidate the proposer already scored internally isn't scored again
    here -- important for a fair "equal number of model calls" comparison
    between proposers with very different per-step costs.
    Returns (trace, accepted): accepted[m] is True iff the proposal at
    step m+1 was accepted -- distinct from trace[m] != trace[m-1], since
    an accepted proposal can re-propose the current state (e.g. an
    independence sampler that isn't restricted to propose a different
    state)."""
    rng      = random.Random(seed)
    current  = initial_state
    lw       = log_weight(text, current, cache=cache)
    trace    = []
    accepted = []
    start    = time.time()
    for step in range(n_steps):
        candidate, log_hastings = proposer(current, rng)
        lw2 = log_weight(text, candidate, cache=cache)       # 1 Qwen call (PyTorch inside)
        accept = math.log(rng.random()) < (lw2 - lw) + log_hastings
        if accept:
            current, lw = candidate, lw2
        trace.append(current)
        accepted.append(accept)
        if progress_every and (step + 1) % progress_every == 0:
            print(f"  step {step + 1}/{n_steps}  ({time.time() - start:.0f}s elapsed)")
    return trace, accepted


def uniform_proposer(idioms):
    n = len(idioms)
    def propose(current, rng):
        while True:
            candidate = idioms[rng.randrange(n)]
            if candidate != current:
                return candidate, 0.0
    return propose


def text_proposer(idioms: list, text: str, static_embeddings: np.ndarray,
                   layer: int, temperature: float = 1.0, standardize: bool = False):
    """Informed proposer, constant cost: a single Qwen forward pass (on the
    text alone, here at construction time), compared against the idioms'
    static embeddings precomputed once (geometry.embeddings). `layer` must
    come from 09_classification_delta.py — do not pick one at random.

    `standardize`: apply per-coordinate standardization (mean/std) to both
    the static embeddings and the text state before computing cosine
    similarity. Off by default for backward compatibility, but this is a
    real fix, not a speculative one: scripts/11_check_hub_idiom.py (Part 3)
    showed one idiom's dominant, text-independent cosine similarity
    (a rogue-dimension artifact, Timkey & van Schijndel 2021) collapse
    entirely once standardized. Requires
    scripts/12_fit_text_state_stats.py to have been run for this layer."""
    v = text_state_by_layer(text)[layer].cpu().numpy()
    if standardize:
        mu_s, sigma_s = static_embedding_stats()
        mu_t, sigma_t = load_text_state_stats(layer)
        static_embeddings = (static_embeddings - mu_s) / sigma_s
        v = (v - mu_t) / sigma_t
    norms = np.maximum(np.linalg.norm(static_embeddings, axis=1) * np.linalg.norm(v), 1e-12)
    weights = np.exp((static_embeddings @ v / norms) / temperature)
    weights /= weights.sum()
    index = {idiom: k for k, idiom in enumerate(idioms)}
    weights_list = weights.tolist()
    # cum_weights precomputed ONCE, not per step -- see delta_proposer_from_scores
    # for why: random.Random.choices rebuilds the whole cumulative-weight
    # table from scratch every call when given weights= instead.
    cum_weights = list(itertools.accumulate(weights_list))

    print(f"informed proposer built: layer {layer}, T={temperature}, "
          f"standardize={standardize}, {len(idioms)} idioms, "
          f"weight min={weights.min():.2e} "
          f"max={weights.max():.2e} (most favored idiom: {idioms[weights.argmax()]})")

    def propose(current, rng):
        idx = rng.choices(range(len(idioms)), cum_weights=cum_weights)[0]
        candidate = idioms[idx]
        log_hastings = math.log(weights_list[index[current]]) - math.log(weights_list[idx])
        return candidate, log_hastings
    return propose


def raw_delta_scores(idioms: list, text: str, layer: int = 23, batch_size: int = 32) -> dict:
    """Delta_layer^E(i,t) (representation.euclidean_delta) for EVERY idiom in
    `idioms` against one fixed text -- the expensive, one-time-per-text pass
    delta_proposer_from_scores' weights are built from (chapter section
    sec:delta_proposal). Batched through context_states_batch, the same
    machinery scripts 17/18 use. Independent of beta: computed once per
    text, then reused for every beta in a sweep, since Delta_l^E(i,t) itself
    does not depend on beta -- only the softmax weighting built from it
    does."""
    scores = {}
    for start in range(0, len(idioms), batch_size):
        batch = idioms[start:start + batch_size]
        pairs = [(idiom, text) for idiom in batch]
        states = context_states_batch(pairs)   # (batch, n_layers, dim)
        for k, idiom in enumerate(batch):
            scores[idiom] = euclidean_delta(states[k, 0], states[k, layer])
    return scores


def delta_proposer_from_scores(idioms: list, delta_scores: dict, beta: float):
    """Independence sampler q_beta(i|t) = softmax(-beta * Delta_l^E(i,t))
    (chapter section sec:delta_proposal), built from ALREADY-COMPUTED scores
    (raw_delta_scores) -- no new model calls here, so sweeping several beta
    values for the same text only pays the dictionary-wide pass once.
    beta=0 recovers uniform_proposer exactly (every weight equal), as a
    mathematical consequence of the softmax at beta=0, not a special case
    handled separately.

    Same shape as text_proposer: an independence sampler, so log_hastings is
    the log ratio of the (state-independent) proposal weights, not of a
    state-conditional distribution."""
    raw = np.array([delta_scores[i] for i in idioms])
    # subtract the min before exponentiating -- numerical stability only,
    # cancels exactly in the softmax (same log-sum-exp reasoning used
    # elsewhere in this project), does not change q_beta itself
    weights = np.exp(-beta * (raw - raw.min()))
    weights /= weights.sum()
    index = {idiom: k for k, idiom in enumerate(idioms)}
    weights_list = weights.tolist()
    # cum_weights precomputed ONCE here, not per step: random.Random.choices
    # rebuilds the entire cumulative-weight table from scratch every call
    # when given weights= (confirmed by direct timing: ~0.5ms/call at
    # |idioms|=31,113, vs ~0.001ms/call passing cum_weights= instead) --
    # a real, model-independent bottleneck once N_STEPS is scaled up, since
    # this same distribution is sampled from unchanged at every step.
    cum_weights = list(itertools.accumulate(weights_list))

    def propose(current, rng):
        idx = rng.choices(range(len(idioms)), cum_weights=cum_weights)[0]
        candidate = idioms[idx]
        log_hastings = math.log(weights_list[index[current]]) - math.log(weights_list[idx])
        return candidate, log_hastings
    return propose
