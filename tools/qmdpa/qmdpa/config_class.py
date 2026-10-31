#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulation
# SPDX-License-Identifier: GPL-2.0-only
"""Configuration classes and the conditional-cost decomposition of V2-11.

For the reactant reference ensemble the roadmaps define the strict probability
identity

    P_R(G_j and X_eps) = p_j * P_R(X_eps | G_j)

    -kT ln P_R(G_j and X_eps)
        = -kT ln p_j - kT ln P_R(X_eps | G_j)                    (V2-11)

where ``X_eps = {|xi| < eps}`` collects the near-degenerate configurations and
``G_j`` partitions the reactant ensemble into disjoint configuration classes.
This module computes both terms from an accepted trajectory, together with the
joint probability, so that a report can be recomputed from raw samples.

Two disciplines are enforced here. First, the estimator is a plain binomial
count on the declared reactant ensemble: the module never reweights, never
interpolates and never smooths across class boundaries. Second, the identity is
checked against the joint count rather than asserted, so that an inconsistent
partition or a truncated support domain is reported instead of hidden.
"""

import math

# Boltzmann constant in eV/K. The identity below is a pure probability
# identity, so the constant only converts the log probabilities into energy
# units and never changes the reported probabilities.
KB_EV_PER_K = 8.617333262e-5

FORM = "configuration-class-decomposition-v1"


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    return float(value)


def partition_counts(samples, classes, gap_window):
    """Count the reactant ensemble into disjoint configuration classes.

    Parameters
    ----------
    samples:
        Iterable of ``(gap_eV, class_id)`` pairs. ``gap_eV`` is the signed
        constrained-state gap of one accepted frame and ``class_id`` names the
        configuration class that the frame belongs to.
    classes:
        The declared class identifiers. Every class must receive at least one
        sample, otherwise the class probability is undefined and the
        decomposition cannot be reported.
    gap_window:
        The half-width ``eps`` of the near-degenerate window ``X_eps``.

    Returns
    -------
    dict
        Per-class sample counts, the near-degenerate count inside each class
        and the total sample count.
    """
    eps = _finite(gap_window, "gap_window")
    if eps <= 0:
        raise ValueError("gap_window must be positive")
    declared = list(classes)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("classes must be a nonempty list of distinct identifiers")
    counts = {name: 0 for name in declared}
    near = {name: 0 for name in declared}
    total = 0
    for index, item in enumerate(samples):
        try:
            gap, name = item
        except (TypeError, ValueError) as exc:
            raise ValueError("Each sample must be a (gap_eV, class_id) pair") from exc
        gap = _finite(gap, "sample gap")
        if name not in counts:
            raise ValueError("Sample refers to an undeclared configuration class")
        counts[name] += 1
        if abs(gap) < eps:
            near[name] += 1
        total += 1
    if total == 0:
        raise ValueError("The reactant ensemble is empty")
    empty = sorted(name for name in declared if counts[name] == 0)
    if empty:
        raise ValueError("Configuration classes without samples have undefined probability: "
                         + ", ".join(map(str, empty)))
    return {"counts": counts, "near": near, "total": total}


def wilson_interval(successes, trials, z=1.96):
    """Wilson score interval for a binomial proportion.

    The Wilson interval is used rather than the normal approximation because
    the near-degenerate counts are routinely small, where the normal
    approximation is invalid and produces intervals outside [0, 1].
    """
    successes, trials = int(successes), int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    z = _finite(z, "z")
    if z <= 0:
        raise ValueError("z must be positive")
    n = float(trials)
    p = successes/n
    denominator = 1.0+z*z/n
    center = (p+z*z/(2*n))/denominator
    spread = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0.0, center-spread), min(1.0, center+spread)


def decompose(samples, classes, gap_window, *, temperature_K=298.15, z=1.96):
    """Compute the V2-11 conditional-cost decomposition.

    Returns the per-class probability cost ``-kT ln p_j``, the conditional cost
    ``-kT ln P_R(X_eps | G_j)`` and their sum, which must equal the joint
    cost ``-kT ln P_R(G_j and X_eps)``. The equality is verified numerically
    and a mismatch is raised, because a mismatch means the partition or the
    support domain is inconsistent.
    """
    temperature_K = _finite(temperature_K, "temperature_K")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    kT = KB_EV_PER_K*temperature_K
    tally = partition_counts(samples, classes, gap_window)
    total = tally["total"]
    rows = []
    for name in tally["counts"]:
        trials = tally["counts"][name]
        hits = tally["near"][name]
        p_j = trials/total
        joint = hits/total
        # The conditional probability is the joint probability divided by the
        # class probability. When a class has no near-degenerate member the
        # conditional probability is zero and its log is undefined; such a
        # class is reported explicitly rather than silently dropped.
        conditional = joint/p_j
        p_low, p_high = wilson_interval(hits, trials, z)
        row = {"class_id": name,
               "samples": trials,
               "near_degenerate_samples": hits,
               "p_j": p_j,
               "p_j_interval": [p_low, p_high],
               "joint_probability": joint,
               "conditional_probability": conditional if conditional > 0 else None,
               "configuration_cost_eV": -kT*math.log(p_j),
               "conditional_cost_eV": (-kT*math.log(conditional)) if conditional > 0 else None}
        if conditional > 0:
            # The identity must hold exactly for the same ensemble and binning.
            joint_cost = -kT*math.log(joint)
            total_cost = row["configuration_cost_eV"]+row["conditional_cost_eV"]
            if abs(total_cost-joint_cost) > 1e-9:
                raise ValueError("The V2-11 decomposition identity failed for class "
                                 + str(name))
            row["joint_cost_eV"] = joint_cost
            row["identity_residual_eV"] = total_cost-joint_cost
        else:
            row["joint_cost_eV"] = None
            row["identity_residual_eV"] = None
        rows.append(row)
    return {"form": FORM,
            "gap_window_eV": _finite(gap_window, "gap_window"),
            "temperature_K": temperature_K,
            "kT_eV": kT,
            "total_samples": total,
            "classes": rows}


def effective_samples(samples, classes, gap_window):
    """Report the near-degenerate sample count per class.

    The effective sample size of the conditional probability is the number of
    samples in the class that also fall inside the near-degenerate window. A
    class with very few such samples gives a wide interval and must be
    reported as such rather than quoted as a precise conditional cost.
    """
    tally = partition_counts(samples, classes, gap_window)
    return {"total_samples": tally["total"],
            "per_class": {name: {"samples": tally["counts"][name],
                                  "near_degenerate_samples": tally["near"][name]}
                           for name in tally["counts"]}}
