.. index:: fix pimd/adaptive

fix pimd/adaptive command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID all pimd/adaptive temp T center X Y Z core Rcore outer Router initialize auxiliary keyword values ...

* T = temperature defining the path springs (temperature units)
* X,Y,Z = fixed reference point (distance units)
* alternatively, *center group group-ID* uses the mass-weighted center of a static reference group
* Rcore,Router = full-quantum and zero-amplitude radii (distance units)
* optional keyword = *sp*, *thermostat*, *geometry*, or *quantum*
* *quantum* value = group-ID of atoms kept at unit path amplitude
* *geometry* values = sphere or slab axis (axis = x, y, or z)
* *sp* value = positive multiplier for Planck's constant (default 1)
* *thermostat* values = pile tau seed
* tau = centroid thermostat time (time units)
* seed = positive random seed

Examples
""""""""

.. code-block:: LAMMPS

   fix pi all pimd/adaptive temp 300 center 10 10 10 core 3 outer 6 initialize auxiliary
   fix pi all pimd/adaptive temp 300 center 10 10 10 core 3 outer 6 initialize auxiliary thermostat pile 0.1 19771

Description
"""""""""""

.. versionadded:: TBD

Sample a spatially varying path-integral extended ensemble. Every partition
retains one auxiliary bead for every atom. The physical position supplied to
the pair potential is

.. math::

   q_{bi}=c_i+a_i(c_i)(y_{bi}-c_i),\qquad c_i=P^{-1}\sum_b y_{bi}.

The amplitude is one inside *core*, zero outside *outer*, and a quintic
switch with continuous first and second derivatives between them. Every
species uses the same amplitude law and its own physical mass. Selection
depends on the atom centroid, not a fixed molecule label.

.. versionadded:: TBD

The default *geometry sphere* measures the distance to the reference point.
With *geometry slab z*, the distance is the absolute z separation; the x and y
coordinates do not change the amplitude. The other slab axes work similarly.
The *core* and *outer* values are half-widths for a slab. The normal is fixed in
Cartesian space.

With *center group group-ID*, the reference is the mass-weighted center of the
auxiliary centroids in that group. Membership is fixed; surrounding atoms enter
and leave the region continuously. The reference group must be nonempty, identical
in all partitions, and spatially compact within every periodic half-box. Its
lowest-ID atom sets the periodic image convention. Relative velocities and the
reaction forces on reference atoms include all derivatives of this moving center.
A group may represent stable solute atoms or a catalytic site; do not dynamically
reassign it to a different defect. Changing its membership after initialization
or across a restart is rejected.

.. code-block:: LAMMPS

   group site id 1 2 3
   group solute id 1 2 3
   fix pi all pimd/adaptive temp 300 center group site core 3 outer 6 quantum solute initialize auxiliary
   fix pi all pimd/adaptive temp 300 center 0 0 10 geometry slab z core 2 outer 4 initialize auxiliary


.. versionadded:: TBD

The optional *quantum group-ID* keeps every atom in that static group at amplitude
one, regardless of its position. This can protect a complete stable solute or a
user-selected set of fragments. It does not freeze their positions. The surrounding
atoms still follow the smooth spatial resolution. No bonds are removed from the
potential and no molecular connectivity is inferred. Group membership must remain
nonempty, fixed and identical across partitions and restarts. For reactive solvents,
use the spatial buffer and coverage diagnostics; do not repeatedly reassign a
transferring proton to a different static group. A common position-dependent
amplitude for an arbitrary moving fragment is not supplied by this option.

The configurational potential in the auxiliary coordinates is

.. math::

   U=\frac{1}{P}\sum_b E(q_b)
     +\frac{P}{2(\beta\hbar)^2}\sum_{i,b}m_i|y_{b+1,i}-y_{bi}|^2.

The integration uses the equivalent
P-times-energy, P-times-temperature convention. The auxiliary free Gaussian
normalization is independent of position. No additional logarithmic amplitude
term is applied. At amplitude zero the physical positions coincide and the
auxiliary internal modes decouple; at amplitude one this is ordinary finite-P
PIMD. The transition force includes the amplitude derivative.

This definition is an explicit adaptive sampling ensemble. An interacting
fluid can have a quantum/classical free-energy mismatch across the buffer.
It does not by itself guarantee the full-quantum core distribution.
Compensation and region-size convergence must be assessed for the target
system. No force-evaluation reduction is performed: all P complete
configurations are evaluated each step. Trajectories are sampling trajectories,
not physical quantum real-time dynamics.

Initialization, output, and restart
"""""""""""""""""""""""""""""""""""

The mandatory *initialize auxiliary* keyword means initial coordinates and
velocities describe y and its time derivative. They are mapped before the first
force calculation. Auxiliary paths must be consistently unwrapped across
partitions; no physical-to-auxiliary inversion is performed. In particular,
collapsed physical coordinates cannot determine nonzero hidden modes.

Changing physical coordinates or velocities outside this integrator after its first run is rejected. Start a new fix with an explicit auxiliary state when a new initialization is intended.

Atom coordinates and velocities exposed to dumps are q and its time derivative.
Atom forces retain the physical-potential forces plus P times the normalized physical bias gradient, when present; the fix columns report their auxiliary pullback. The fix publishes
13 per-atom columns: auxiliary positions (1-3), auxiliary velocities (4-6),
conjugate dynamical forces (7-9), amplitude (10), and its gradient (11-13).
Springs are propagated separately. These fictitious velocities must not be
interpreted as physical thermal velocities.

The scalar is extended total energy. Five extensive vector entries are
mean physical potential, auxiliary spring energy, auxiliary kinetic energy
divided by P, extended total energy, and physical bias energy. The total includes the bias. These are not quantum thermodynamic estimators.
Ordinary thermo temperature from physical velocities is not the thermostat
temperature diagnostic.

Restart preserves all auxiliary state and the optional PILE random stream.
Use the same fix ID and settings when reading a restart. Exact stochastic
continuation additionally requires the same processor layout and atom order,
as with other per-rank random streams. Use atom sorting disabled for that check.

Restrictions
""""""""""""

Requires REPLICA, group all, identical atom IDs/types/masses across partitions,
equal processor counts, a power-of-two partition count, fixed atom count,
an atom map, consecutive atom IDs, Verlet, and lj, real or metal units.
The simulation cell must be fixed and orthorhombic without shrink-wrap boundaries.
The outer radius must be smaller than each relevant periodic half-box
(all axes for a sphere, only the normal axis for a slab). Periodic
centroid-to-anchor distances use minimum images.

One registered PIMD bias adapter, print and ave/time companion fixes are accepted.
With :doc:`fix plumed <fix_plumed>`, define this integrator first and select
coordinate_mean, cv_mean or bias_mean explicitly. With cv_mean, the
:doc:`path_contraction option <fix_plumed>` is available in this fixed-cell
integrator. It contracts the physical paths before evaluating PLUMED and pulls
the gradients back through both maps. It does not reduce the number of potential
evaluations. Every action in that PLUMED instance sees the contracted coordinates;
this includes walls and region definitions. A physical-bead confinement wall must
use an uncontracted view. Geometry contraction and probability-mixture coupling
are different parameters with different permitted ranges. The physical bias gradient
is pulled back through the same auxiliary map exactly once, including forces
on classical-environment atoms. A zero CV weight does not prevent exterior
reactions or confine a defect.

The bias virial includes the derivative of the amplitude, with either a fixed
Cartesian reference or an affinely moving reference group. This derivative is not an NPT or thermodynamic-pressure qualification.
Barostats, constraints, atom insertion/deletion and arbitrary force-modifying
fixes are not supported. Existing fixed mixed-bead
:doc:`fix pimd/mixed <fix_pimd_mixed>` retains its separate behavior.

Default
"""""""

No thermostat, sp = 1, and geometry sphere.

Execution profiling
-------------------

Set LAMMPS_PIMD_ADAPTIVE_PROFILE=1 before creating the fix to collect
optional inclusive wall-clock timings without adding MPI barriers or GPU
synchronization. The developer extract("pimd_adaptive_profile", dim)
interface returns 24 doubles as (seconds, calls) pairs for: inter-replica
collection, transform arithmetic, position mapping, velocity mapping,
force pullback, initial integration, final integration, end-of-step
estimators, potential-view preparation, kick, free propagation, and
thermostat. Values accumulate over the fix lifetime, including setup and
warmup. Nested phases overlap; do not sum them or sum waiting times over
ranks. Subtract snapshots to isolate a chosen interval. Timing is disabled
by default and does not form part of restart state.
