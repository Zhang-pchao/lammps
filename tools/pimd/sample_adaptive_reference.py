#!/usr/bin/env python3
"""Finite-time BAOAB validation against an independent harmonic marginal.

This samples the declared extended ensemble. It is not a production integrator,
water benchmark, quantum transport simulation, or proof of fluid compensation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from adaptive_reference import AdaptivePathMeasure, SphereResolution, harmonic_potential


def target_moments():
    x = np.linspace(-6, 6, 24001)
    t = np.clip((np.abs(x) - 0.4) / 0.8, 0, 1)
    a = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
    beta, p, k = 2.0, 4, 1.0
    # Direct cyclic eigenvalues, independently of the implementation eigensolver.
    kappa = p * 4 * np.sin(np.pi * np.arange(1, p) / p) ** 2 / (beta * 2.0) ** 2
    free = 0.5 * x**2 + np.log1p(k * a[:, None] ** 2 / (p * kappa)).sum(1) / (2 * beta)
    density = np.exp(-beta * free)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    density /= integrate(density, x)
    rg2 = a**2 * np.sum(1 / (beta * (kappa + k * a[:, None] ** 2 / p)), axis=1) / p
    observables = [x * x, x**4, (np.abs(x) < 0.4).astype(float), x * x + rg2, rg2]
    return dict(
        zip(
            ("centroid2", "centroid4", "core_probability", "physical2", "radius2"),
            (float(integrate(value * density, x)) for value in observables),
        )
    )


def sample(dt, steps, seed, omit_transition=False):
    beta, chains, burn = 2.0, 32, steps // 10
    rng = np.random.default_rng(seed)
    model = AdaptivePathMeasure(4, [1.0], beta, 2.0, SphereResolution((0.0,), 0.4, 1.2))
    potential = harmonic_potential()
    c = rng.normal(scale=0.7, size=(chains, 1, 1))
    z = rng.normal(size=(chains, 3, 1, 1)) / np.sqrt(beta * model.stiffness[..., None])
    mc = np.ones_like(c)
    mz = np.broadcast_to(model.stiffness[..., None], z.shape)
    vc = rng.normal(size=c.shape) / np.sqrt(beta * mc)
    vz = rng.normal(size=z.shape) / np.sqrt(beta * mz)
    decay = np.exp(-dt)
    noise_c, noise_z = (
        np.sqrt((1 - decay**2) / (beta * mc)),
        np.sqrt((1 - decay**2) / (beta * mz)),
    )
    sums, count = np.zeros((chains, 5)), 0

    def forces():
        _, fc, fz, _ = model.evaluate(c, z, potential)
        if omit_transition:
            fc = potential(model.positions(c, z))[1].mean(axis=-3)
        return fc, fz

    fc, fz = forces()
    for step in range(steps):
        vc += 0.5 * dt * fc / mc
        vz += 0.5 * dt * fz / mz
        c += 0.5 * dt * vc
        z += 0.5 * dt * vz
        vc = decay * vc + noise_c * rng.normal(size=c.shape)
        vz = decay * vz + noise_z * rng.normal(size=z.shape)
        c += 0.5 * dt * vc
        z += 0.5 * dt * vz
        fc, fz = forces()
        vc += 0.5 * dt * fc / mc
        vz += 0.5 * dt * fz / mz
        if step >= burn and step % 10 == 0:
            q = model.positions(c, z)
            x = c[:, 0, 0]
            rg2 = ((q - c[:, None]) ** 2).mean(axis=(1, 2, 3))
            sums += np.column_stack(
                (x * x, x**4, np.abs(x) < 0.4, (q * q).mean(axis=(1, 2, 3)), rg2)
            )
            count += 1
    replica_means = sums / count
    reference = target_moments()
    checks = []
    for i, (name, expected) in enumerate(reference.items()):
        mean = float(replica_means[:, i].mean())
        se = float(replica_means[:, i].std(ddof=1) / np.sqrt(chains))
        # Six between-replica standard errors plus a predeclared 0.5% scale floor.
        tolerance = 6 * se + 0.005 * max(abs(expected), 0.1)
        checks.append(
            {
                "observable": name,
                "measured": mean,
                "reference": expected,
                "replica_se": se,
                "tolerance": tolerance,
                "passed": bool(abs(mean - expected) <= tolerance),
            }
        )
    return {
        "dt": dt,
        "steps": steps,
        "chains": chains,
        "saved_per_chain": count,
        "seed": seed,
        "omitted_transition": omit_transition,
        "checks": checks,
        "replica_means": replica_means.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    if args.steps < 10000:
        parser.error("At least 10000 steps are required for this statistical pilot")
    report = {
        "scope": "Finite-time harmonic extended-ensemble pilot",
        "passed": False,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dt, seed, negative in (
        (0.04, 7311, False),
        (0.02, 7312, False),
        (0.04, 7313, True),
    ):
        report["runs"].append(sample(dt, args.steps, seed, negative))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    positive = all(c["passed"] for run in report["runs"][:2] for c in run["checks"])
    negative_detected = not report["runs"][2]["checks"][0]["passed"]
    report["negative_control_detected"] = negative_detected
    report["passed"] = positive and negative_detected
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": report["passed"], "negative_control_detected": negative_detected}
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
