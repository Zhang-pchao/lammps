# Adaptive path measure reference

This is a small method-validation tool, not an implemented LAMMPS fix. It uses
NumPy only. Run `python -m unittest discover -s tools/pimd -p 'test_*.py'`.

## Explicit configurational ensemble

Let c_i be the centroid, z_ki internal orthonormal normal-mode coordinates, T_bk
the cyclic spring eigenvectors without the centroid, and u_bi = sum_k T_bk z_ki.
For a dimensionless smooth spatial amplitude a_i(c_i), define

    q_bi = c_i + a_i(c_i) u_bi
    U(c,z) = (1/P) sum_b E(q_b) + (1/2) sum_ki kappa_ki |z_ki|^2
    kappa_ki = m_i P lambda_k / (beta hbar)^2
    rho(c,z) dc dz proportional to exp[-beta U(c,z)] dc dz.

The normalized free internal Gaussian is independent of the amplitude. For a>0,
changing variables to physical internal coordinates s=a z gives the potential
kappa |s|^2/(2 a^2) plus D(P-1) log(a)/beta for each atom. Its Jacobian is therefore
already included by sampling (c,z); never add another log-amplitude correction.
At a=0 all physical beads coincide while the auxiliary modes remain finite and
decoupled. This endpoint has exactly classical configurational statistics, not
one dynamically inserted/deleted bead. At a=1 the measure is ordinary finite-P
PIMD, up to the constant centroid-coordinate Jacobian.

For physical forces f_bi = -dE/dq_bi, the conjugate forces are

    F_ci = mean_b f_bi + grad(a_i) mean_b (f_bi dot u_bi)
    F_zki = a_i/P sum_b T_bk f_bi - kappa_ki z_ki.

A translated anchor receives the opposite sum of the transition terms.
Fictitious momentum masses may be chosen as positive constants; they do not
alter the above configurational measure. True elemental masses stay in kappa.
No discrete molecule identity, molecular pairing, or species-specific amplitude
is assumed. No saved prototype trajectory is physical quantum real-time motion.

## What this does and does not establish

This is an explicit extended sampling model, with exact finite-P and classical
endpoints. It is a centroid-based adaptive definition, not a derivation that
spatially varying nuclear masses describe the original physical Hamiltonian.
PI-AdResS is a relevant precedent (Kreis et al., JCP 147, 244104, 2017,
https://doi.org/10.1063/1.5000701); this tool is an independent normalized-coordinate
prototype, not a port or a reproduction of that implementation.

A free particle has a uniform centroid density without an amplitude-dependent
entropic drift. An interacting system generally does not: integrating the
physical potential over internal modes produces a position-dependent quantum
free-energy contribution. Compensation, buffer/region convergence and the target
core ensemble must be established before claiming full-quantum core statistics.
A generic compensation fitted to water is not assumed transferable to a surface.

For a 1D harmonic well k c^2/2, conditional integration is analytic:

    F_eff(c) = k c^2/2 + sum_k log[1 + k a(c)^2/(P kappa_k)]/(2 beta).

This expression is an independent distribution oracle for the adaptive pilot.
The all-quantum position variance is compared to the cyclic bead-Hessian inverse;
spatial forces are compared to finite differences including an asymmetric
many-body potential and a moving anchor. A negative control removes the
transition force and must fail those derivative/distribution checks.

P full-system potential evaluations remain. This tool provides no PES locality,
RPC, long-range compatibility, production compensation or acceleration claim.
Native integration, biases, restart, MPI and GPU need their own implementation
and admission. Current geometry is nonperiodic with a fixed sphere anchor; no
periodic centroid-unwrapping rule is silently assumed.
