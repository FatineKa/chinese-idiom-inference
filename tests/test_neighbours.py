"""Tests for chengyu/neighbours.py.
- numpy only: no model, no data files.
- Run in CI on every push."""
import numpy as np

from chengyu.neighbours import combine, idioms_from_neighbours


def test_neighbours_best_first_without_repeats():
    example_idioms = ["甲", "乙", "甲", "丙"]
    sim = np.array([0.9, 0.5, 0.8, 0.1])
    assert idioms_from_neighbours(sim, example_idioms, 3) == ["甲", "乙", "丙"]


def test_neighbours_stops_at_k():
    sim = np.array([0.1, 0.9, 0.5])
    assert idioms_from_neighbours(sim, ["甲", "乙", "丙"], 1) == ["乙"]


def test_combine_takes_in_turn():
    assert combine(["a1", "a2"], ["b1", "b2"], 4) == ["a1", "b1", "a2", "b2"]


def test_combine_skips_repeats_and_respects_k():
    assert combine(["x", "a"], ["x", "b"], 2) == ["x", "a"]


def test_combine_uses_leftovers():
    assert combine(["a1", "a2", "a3"], ["b1"], 4) == ["a1", "b1", "a2", "a3"]
