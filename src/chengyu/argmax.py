"""argmax.py — exact brute-force ranking of idioms.
Replaces MCMC: for the argmax, the constant Z is irrelevant
(it's the same for every idiom), so we score the whole
dictionary and take the max."""
import math

import torch

from chengyu.prior import log_prior
from chengyu.scoring import _device, _model, _tok


@torch.no_grad()
def text_scores(text: str, idioms: list, batch_size: int = 16, cache: dict | None = None) -> dict:
    """log p(text | idiom) for each idiom, in batches.
    A single forward pass per prompt+text: we sum the log-probabilities of
    the text's own tokens only — equivalent to the subtraction
    log p(prompt+text) - log p(prompt) via the chain rule, but half as
    expensive.
    cache: optional {(text, idiom): score} dict, reused across calls (e.g.
    across different values of k) so each (text, idiom) pair is scored once."""
    if cache is None:
        cache = {}
    to_score = [i for i in idioms if (text, i) not in cache]
    text_ids = _tok(text, add_special_tokens=False).input_ids
    pad = _tok.pad_token_id or _tok.eos_token_id
    for start in range(0, len(to_score), batch_size):
        batch = to_score[start:start + batch_size]
        sequences, prompt_lengths = [], []
        for idiom in batch:
            prompt_ids = _tok(f"成语「{idiom}」概括了这句话：").input_ids
            prompt_lengths.append(len(prompt_ids))
            sequences.append(prompt_ids + text_ids)
        # right padding: no effect on the positions that matter (causal model)
        L = max(len(s) for s in sequences)
        input_ids = torch.full((len(batch), L), pad, dtype=torch.long, device=_device)
        mask = torch.zeros((len(batch), L), dtype=torch.long, device=_device)
        for j, s in enumerate(sequences):
            input_ids[j, :len(s)] = torch.tensor(s, device=_device)
            mask[j, :len(s)] = 1
        logp = torch.log_softmax(_model(input_ids, attention_mask=mask).logits, dim=-1)
        for j, idiom in enumerate(batch):
            total = 0.0
            for k in range(prompt_lengths[j], len(sequences[j])):   # text tokens only
                total += logp[j, k - 1, sequences[j][k]].item()
            cache[(text, idiom)] = total
    return {i: cache[(text, i)] for i in idioms}


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
