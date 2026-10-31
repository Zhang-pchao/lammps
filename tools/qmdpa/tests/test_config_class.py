#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulation
# SPDX-License-Identifier: GPL-2.0-only
"""Configuration-class decomposition and the V2-11 identity controls."""

import math

import pytest

from qmdpa.config_class import (FORM, KB_EV_PER_K, decompose, effective_samples,
                                partition_counts, wilson_interval)


def samples():
    """A small reactant ensemble with two disjoint configuration classes.

    Class ``close`` holds the near-degenerate configurations, class ``far``
    holds only well-separated ones.
    """
    return [(-0.10, "close"), (0.05, "close"), (-0.20, "close"), (0.15, "close"),
            (0.90, "far"), (1.20, "far"), (-1.10, "far"), (0.80, "far")]


def test_partition_counts_cover_the_declared_classes():
    tally = partition_counts(samples(), ["close", "far"], 0.3)
    assert tally["total"] == 8
    assert tally["counts"] == {"close": 4, "far": 4}
    assert tally["near"] == {"close": 4, "far": 0}


def test_partition_rejects_an_undeclared_class():
    with pytest.raises(ValueError, match="undeclared configuration class"):
        partition_counts([(0.1, "other")], ["close"], 0.3)


def test_partition_rejects_a_class_without_samples():
    with pytest.raises(ValueError, match="undefined probability"):
        partition_counts([(0.1, "close")], ["close", "far"], 0.3)


def test_partition_rejects_an_empty_ensemble():
    with pytest.raises(ValueError, match="empty"):
        partition_counts([], ["close"], 0.3)


def test_partition_rejects_a_nonpositive_window():
    with pytest.raises(ValueError, match="must be positive"):
        partition_counts(samples(), ["close", "far"], 0.)


def test_partition_rejects_a_nonfinite_sample():
    with pytest.raises(ValueError, match="finite"):
        partition_counts([(float("nan"), "close")], ["close"], 0.3)


def test_wilson_interval_stays_inside_the_unit_interval():
    low, high = wilson_interval(1, 10)
    assert 0. <= low < 0.1 < high <= 1.
    low, high = wilson_interval(10, 10)
    assert 0. <= low <= 1. and high == 1.
    low, high = wilson_interval(0, 10)
    assert low == 0. and high < 1.


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_decomposition_reproduces_the_v2_11_identity_exactly():
    report = decompose(samples(), ["close", "far"], 0.3)
    assert report["form"] == FORM
    kT = KB_EV_PER_K*298.15
    rows = {row["class_id"]: row for row in report["classes"]}
    close = rows["close"]
    # p_j = 4/8, P(X_eps | G_j) = 4/4, joint = 4/8.
    assert close["p_j"] == pytest.approx(0.5)
    assert close["conditional_probability"] == pytest.approx(1.0)
    assert close["configuration_cost_eV"] == pytest.approx(-kT*math.log(0.5))
    assert close["conditional_cost_eV"] == pytest.approx(0.0)
    assert close["joint_cost_eV"] == pytest.approx(-kT*math.log(0.5))
    assert close["identity_residual_eV"] == pytest.approx(0.0, abs=1e-12)


def test_decomposition_reports_a_class_without_near_degenerate_members():
    report = decompose(samples(), ["close", "far"], 0.3)
    rows = {row["class_id"]: row for row in report["classes"]}
    far = rows["far"]
    assert far["near_degenerate_samples"] == 0
    assert far["conditional_probability"] is None
    assert far["conditional_cost_eV"] is None
    assert far["joint_cost_eV"] is None
    # The configuration cost is still well defined.
    assert far["configuration_cost_eV"] > 0.


def test_decomposition_probabilities_are_consistent_with_the_joint():
    report = decompose(samples(), ["close", "far"], 0.3)
    for row in report["classes"]:
        assert row["p_j"] == pytest.approx(row["samples"]/report["total_samples"])
        assert row["joint_probability"] == pytest.approx(
            row["near_degenerate_samples"]/report["total_samples"])
        if row["conditional_probability"] is not None:
            assert row["joint_probability"] == pytest.approx(
                row["p_j"]*row["conditional_probability"])


def test_decomposition_reports_the_effective_sample_size():
    report = effective_samples(samples(), ["close", "far"], 0.3)
    assert report["total_samples"] == 8
    assert report["per_class"]["close"]["near_degenerate_samples"] == 4
    assert report["per_class"]["far"]["near_degenerate_samples"] == 0


def test_decomposition_window_selects_the_near_degenerate_frames():
    narrow = decompose(samples(), ["close", "far"], 0.15)
    rows = {row["class_id"]: row for row in narrow["classes"]}
    assert rows["close"]["near_degenerate_samples"] == 2


def test_decomposition_is_reproducible_from_the_same_samples():
    first = decompose(samples(), ["close", "far"], 0.3)
    second = decompose(samples(), ["close", "far"], 0.3)
    assert first == second


def test_decomposition_rejects_a_nonpositive_temperature():
    with pytest.raises(ValueError, match="must be positive"):
        decompose(samples(), ["close", "far"], 0.3, temperature_K=0.)


def test_decomposition_rejects_duplicate_class_identifiers():
    with pytest.raises(ValueError, match="distinct"):
        decompose(samples(), ["close", "close"], 0.3)
