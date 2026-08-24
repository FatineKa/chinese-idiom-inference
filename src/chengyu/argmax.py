"""argmax.py — exact brute-force ranking of idioms.
Replaces MCMC: for the argmax, the constant Z is irrelevant
(it's the same for every idiom), so we score the whole
dictionary and take the max."""
import math

from chengyu.prior import log_prior
from chengyu.scoring import text_scores   # canonical implementation lives in scoring.py;
                                           # re-exported here so `from chengyu.argmax
                                           # import text_scores` keeps working unchanged


def exact_posterior(scores: dict) -> dict:
    """Normalize unnormalised log-scores h_t(i) into pi(i|t) = exp(h_t(i)) / Z_t,
    via the numerically stable log-sum-exp trick (subtract the max before
    exponentiating, so every exponent is <= 0 and overflow is avoided)."""
    m = max(scores.values())
    log_z = m + math.log(sum(math.exp(v - m) for v in scores.values()))
    return {i: math.exp(v - log_z) for i, v in scores.items()}


def posterior_by_length(posterior: dict) -> dict:
    """P(length(I) = n | T = t): group a posterior {idiom: probability}
    by character count and sum the probability within each group.
    No candidate restriction needed -- this just regroups probability
    mass that's already been computed over the whole dictionary."""
    by_length: dict = {}
    for idiom, p in posterior.items():
        n = len(idiom)
        by_length[n] = by_length.get(n, 0.0) + p
    return by_length


def rank(text, idioms, k=10, with_prior=True, batch_size=16, cache=None):
    """Exact top-k: argmax of log p(t|i) (+ log p(i))."""
    scores = text_scores(text, idioms, batch_size, cache=cache)
    if with_prior:
        scores = {i: v + log_prior(i) for i, v in scores.items()}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
