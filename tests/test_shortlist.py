"""Tests for chengyu/shortlist.py.
- numpy only: no model, no data files.
- Run in CI on every push."""
import numpy as np

from chengyu.shortlist import cosine_to_all, top_k


def test_cosine_of_a_vector_with_itself_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_to_all(v[None, :], v)[0], 1.0)


def test_cosine_ignores_vector_length():
    vectors = np.array([[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    sim = cosine_to_all(vectors, np.array([3.0, 0.0]))
    assert np.allclose(sim, [1.0, 1.0, 0.0])


def test_top_k_returns_best_first():
    idioms = ["甲", "乙", "丙", "丁"]
    sim = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k(sim, idioms, 3) == ["乙", "丁", "丙"]


def test_top_k_larger_than_dictionary_returns_everything():
    idioms = ["甲", "乙"]
    assert sorted(top_k(np.array([0.2, 0.1]), idioms, 10)) == sorted(idioms)


def test_zero_vector_does_not_crash():
    sim = cosine_to_all(np.zeros((2, 3)), np.zeros(3))
    assert np.all(np.isfinite(sim))
