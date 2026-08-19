"""representation.py — modification signal (static vs. contextualized),
layer by layer. A single forward pass computes every layer at once
(output_hidden_states costs nothing extra)."""
import os

import numpy as np
import torch

from chengyu.scoring import _device, _model, _tok


@torch.no_grad()
def last_token_states(prompt: str) -> torch.Tensor:
    """State at the last position of `prompt`, at every layer
    (0 = embeddings, n_layers = last). Shape: (n_layers+1, dim)."""
    ids = _tok(prompt, return_tensors="pt").input_ids.to(_device)
    outputs = _model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in outputs.hidden_states]).float()


def modification_by_layer(idiom: str, text: str) -> list:
    """Delta_l(i,t) for each layer l: distance between layer 0 (the raw
    embedding, before any context mixing) and layer l, at the idiom's own
    last-token position. Reversed prompt (text then idiom) to respect
    causal masking: the idiom must come after the text for its
    representation to be influenced by it (causal model — see the
    chapter, section "The causal constraint"). Comparing layer 0 against
    itself when l=0 makes Delta_0=0 an identity, true for idioms of any
    token length, not a coincidence that depends on how the idiom is
    pooled."""
    prompt = f"这句话「{text}」，可以概括为成语「{idiom}"
    states = last_token_states(prompt)
    baseline = states[0]
    # l=0 is forced to exactly 0.0: cosine_similarity(x,x) is not bit-exact
    # 1.0 in floating point (a sqrt-then-square round-trip), which would
    # otherwise leave a ~1e-7 residual that a downstream classifier can
    # latch onto as spurious signal.
    return [0.0] + [
        1 - torch.nn.functional.cosine_similarity(baseline, e, dim=0).item()
        for e in states[1:]
    ]


def text_state_by_layer(text: str) -> torch.Tensor:
    """State of the text alone (without the idiom), at every layer, at the
    position right before the idiom. A single forward pass, independent of
    the number of idioms — used by the efficient proposer
    (mcmc.text_proposer)."""
    prompt = f"这句话「{text}」，可以概括为成语「"
    return last_token_states(prompt)


def load_text_state_stats(layer: int):
    """Load precomputed per-coordinate mean/std of layer-`layer` text
    states (scripts/12_fit_text_state_stats.py), for standardizing before
    cosine similarity in mcmc.text_proposer -- see
    geometry.static_embedding_stats for why. Precomputed to disk rather
    than computed in-process like the static-embedding stats: each text
    state needs a real forward pass, so estimating this from many texts is
    real GPU cost that should happen once, not on every text_proposer
    call."""
    path = f"data/text_state_stats_layer{layer}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/12_fit_text_state_stats.py "
            f"first (fits every layer in one pass, no CHENGYU_LAYER needed)."
        )
    data = np.load(path)
    return data["mu"], data["sigma"] + 1e-8


@torch.no_grad()
def last_token_states_batch(prompts: list) -> torch.Tensor:
    """Like last_token_states, but for a whole batch of prompts in a single
    forward pass — essential on GPU (many small successive calls under-use
    GPU parallelism; a single call over a batch exploits it).

    LEFT padding (not right as in argmax.py): we need the LAST position (-1)
    of each sequence, which only matches the last real token if the padding
    tokens are added before it, not after — unlike argmax.py, which only
    reads text positions (never the absolute last position) and can
    therefore afford right padding.

    position_ids is computed explicitly from the attention mask (rather than
    left to the model's default): under RoPE, a position shifted by padding
    would silently change the meaning of the causal positions for the real
    tokens — don't rely on an implicit default here."""
    pad = _tok.pad_token_id or _tok.eos_token_id
    ids_list = [_tok(p).input_ids for p in prompts]
    L = max(len(ids) for ids in ids_list)
    batch_size = len(prompts)
    batch = torch.full((batch_size, L), pad, dtype=torch.long, device=_device)
    mask = torch.zeros((batch_size, L), dtype=torch.long, device=_device)
    for j, ids in enumerate(ids_list):
        batch[j, L - len(ids):] = torch.tensor(ids, device=_device)
        mask[j, L - len(ids):] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    outputs = _model(batch, attention_mask=mask, position_ids=position_ids,
                      output_hidden_states=True)
    return torch.stack([h[:, -1] for h in outputs.hidden_states], dim=1).float()


def modification_by_layer_batch(pairs: list) -> list:
    """Like modification_by_layer, but for a list of (idiom, text) pairs at
    once — a single forward pass for the whole batch rather than one per
    pair. Returns a list (one entry per pair) of lists of Delta_l."""
    prompts = [f"这句话「{text}」，可以概括为成语「{idiom}" for idiom, text in pairs]
    states = last_token_states_batch(prompts)   # (batch, n_layers, dim)
    results = []
    for k in range(states.shape[0]):
        baseline = states[k, 0]
        # l=0 forced to exactly 0.0 -- see modification_by_layer.
        results.append([0.0] + [
            1 - torch.nn.functional.cosine_similarity(baseline, states[k, l], dim=0).item()
            for l in range(1, states.shape[1])
        ])
    return results


def context_states_batch(pairs: list) -> np.ndarray:
    """Like modification_by_layer_batch, but returns the RAW per-layer states
    (e^(0)..e^(L), the idiom's own last-token state at every layer) instead
    of collapsing them to cosine -- needed to compute more than one distance
    metric from the same forward pass (representation study, section 6:
    plain cosine, standardized cosine, Euclidean). Shape:
    (n_pairs, n_layers+1, dim)."""
    prompts = [f"这句话「{text}」，可以概括为成语「{idiom}" for idiom, text in pairs]
    return last_token_states_batch(prompts).cpu().numpy()


def cosine_delta(e0: np.ndarray, el: np.ndarray) -> float:
    """Delta_l^cos(i,t) = 1 - cos(e^(0), e^(l)) -- same definition as
    modification_by_layer/_batch, but taking already-computed raw states
    (context_states_batch) instead of running its own forward pass, so all
    three distance metrics can share one batch of model calls."""
    denom = max(float(np.linalg.norm(e0)) * float(np.linalg.norm(el)), 1e-12)
    return 1 - float(np.dot(e0, el)) / denom


def standardized_delta(e0: np.ndarray, el: np.ndarray, mu0: np.ndarray,
                        sigma0: np.ndarray, mu_l: np.ndarray, sigma_l: np.ndarray) -> float:
    """Delta_l^std(i,t) = 1 - cos(z0, zl), z = (e - mu) / sigma -- each side
    standardized with ITS OWN layer's calibration stats
    (load_context_state_stats) before comparing direction, so a handful of
    oversized coordinates can't dominate the cosine (rogue-dimension effect,
    Timkey & van Schijndel 2021)."""
    z0 = (e0 - mu0) / sigma0
    zl = (el - mu_l) / sigma_l
    denom = max(float(np.linalg.norm(z0)) * float(np.linalg.norm(zl)), 1e-12)
    return 1 - float(np.dot(z0, zl)) / denom


def euclidean_delta(e0: np.ndarray, el: np.ndarray) -> float:
    """Delta_l^E(i,t) = ||e^(l) - e^(0)||_2 -- magnitude of the shift, unlike
    the two cosine-based metrics above, which only look at direction."""
    return float(np.linalg.norm(el - e0))


def load_context_state_stats(layer: int):
    """Load precomputed per-coordinate mean/std of layer-`layer` states of
    the idiom ITSELF, in context (scripts/16_fit_context_state_stats.py),
    for standardized_delta. Deliberately not geometry.static_embedding_stats()
    -- that describes a different population (an idiom's characters averaged
    over the whole idiom) than e^(0)/e^(l) here (the idiom's own last-token
    state, extracted from the same forward pass as every other layer)."""
    path = f"data/context_state_stats_layer{layer}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/16_fit_context_state_stats.py "
            f"first (fits every layer in one pass)."
        )
    data = np.load(path)
    return data["mu"], data["sigma"] + 1e-8
