"""Tests for "Exact posterior inference on a finite idiom dictionary"
(sec:direct_application): MAP estimation, top-k ranking, posterior
normalisation (log-sum-exp), and the posterior distribution of idiom
length -- all computed over the whole dictionary I, with no length
restriction applied before inference. Uses the small CPU model (see
chengyu/scoring.py) so this runs fast without a GPU.

Run with:
    CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest tests/test_direct_application.py
"""
import math
import os

os.environ.setdefault("CHENGYU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

import pytest

from chengyu.argmax import exact_posterior, posterior_by_length, rank, text_scores
from chengyu.evaluation import idioms_of_length, load_dictionary
from chengyu.prior import log_prior
from chengyu.scoring import score_summary

TEXT = "他每天早起晚睡，非常努力地工作。"


# ---------------------------------------------------------------------------
# Pure logic checks: no model needed, run against the full I_n.
# ---------------------------------------------------------------------------

def test_prior_is_positive_and_normalized_over_full_dictionary():
    # p(i) is a single global prior over the whole dictionary I, not a
    # separate distribution per length n -- so it sums to 1 over I, and only
    # a fraction of that mass falls on any one length class I_n (see
    # exact_posterior() for the per-n conditioning/renormalization step).
    dictionary, _ = load_dictionary()
    assert all(log_prior(i) > -math.inf for i in dictionary)
    total = sum(math.exp(log_prior(i)) for i in dictionary)
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_idioms_of_length_matches_requested_length():
    dictionary, _ = load_dictionary()
    for n in (4, 8):
        i_n = idioms_of_length(dictionary, n)
        assert len(i_n) > 0
        assert all(len(i) == n for i in i_n)


def test_exact_posterior_normalization_on_synthetic_scores():
    # deliberately large/uneven values, to exercise the log-sum-exp overflow guard
    h = {"a": -5.0, "b": 1000.2, "c": 999.9, "d": -1e6}
    pi = exact_posterior(h)
    assert all(math.isfinite(v) for v in pi.values())
    assert all(v >= 0 for v in pi.values())
    assert math.isclose(sum(pi.values()), 1.0, abs_tol=1e-9)
    assert max(h, key=h.get) == max(pi, key=pi.get)


def test_posterior_by_length_conserves_total_mass_synthetic():
    # fake idioms of different lengths -- grouping by length must not lose
    # or double-count any probability mass
    h = {"aaaa": 2.0, "bbbb": -1.0, "cccccccc": 0.5, "dddddddd": 3.0, "eeeee": -0.5}
    pi = exact_posterior(h)
    grouped = posterior_by_length(pi)
    assert math.isclose(sum(grouped.values()), sum(pi.values()), abs_tol=1e-12)
    assert set(grouped) == {4, 8, 5}
    assert math.isclose(grouped[4], pi["aaaa"] + pi["bbbb"], abs_tol=1e-12)
    assert math.isclose(grouped[8], pi["cccccccc"] + pi["dddddddd"], abs_tol=1e-12)


def test_length_partition_is_exhaustive_over_dictionary():
    # every idiom in the real dictionary falls into exactly one length
    # bucket, and the buckets together cover the whole dictionary
    dictionary, _ = load_dictionary()
    lengths_present = {len(i) for i in dictionary}
    covered = set()
    for n in lengths_present:
        covered |= set(idioms_of_length(dictionary, n))
    assert covered == set(dictionary)


# ---------------------------------------------------------------------------
# Model-based checks: one batched forward pass on a small candidate subset.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_candidates():
    # mixed lengths, not restricted to one I_n -- matches this section's
    # "no restriction before inference" formulation
    dictionary, _ = load_dictionary()
    return (
        idioms_of_length(dictionary, 4)[:4]
        + idioms_of_length(dictionary, 8)[:2]
        + idioms_of_length(dictionary, 5)[:2]
    )


@pytest.fixture(scope="module")
def scored(small_candidates):
    cache = {}
    likelihood = text_scores(TEXT, small_candidates, cache=cache)
    h = {i: likelihood[i] + log_prior(i) for i in small_candidates}
    return likelihood, h, cache


def test_scores_are_finite(scored):
    likelihood, h, _ = scored
    assert all(math.isfinite(v) for v in likelihood.values())
    assert all(math.isfinite(v) for v in h.values())


def test_argmax_of_h_matches_argmax_of_posterior(scored):
    _, h, _ = scored
    pi = exact_posterior(h)
    assert max(h, key=h.get) == max(pi, key=pi.get)


def test_cache_reuse_gives_identical_scores(small_candidates, scored):
    _, _, cache = scored
    before = dict(cache)
    # re-request the same (text, idiom) pairs: must come straight from cache,
    # i.e. produce identical scores without adding new entries
    again = text_scores(TEXT, small_candidates, cache=cache)
    assert cache == before
    for i in small_candidates:
        assert again[i] == before[(TEXT, i)]


def test_changing_k_does_not_change_underlying_scores(small_candidates):
    cache = {}
    top2 = rank(TEXT, small_candidates, k=2, cache=cache)
    top5 = rank(TEXT, small_candidates, k=5, cache=cache)
    assert top2 == top5[:2]


def test_posterior_by_length_matches_full_posterior_on_mixed_subset(scored):
    _, h, _ = scored
    pi = exact_posterior(h)
    grouped = posterior_by_length(pi)
    assert math.isclose(sum(grouped.values()), sum(pi.values()), abs_tol=1e-9)
    assert set(grouped) == {len(i) for i in pi}


def test_batched_and_single_sequence_scoring_agree(small_candidates):
    # cross-checks the text-token mask: argmax.text_scores (batched) vs.
    # scoring.score_summary (single sequence) must apply the same mask
    idiom = small_candidates[0]
    batched = text_scores(TEXT, [idiom])[idiom]
    single = score_summary(TEXT, idiom)
    assert math.isclose(batched, single, rel_tol=1e-4, abs_tol=1e-3)
