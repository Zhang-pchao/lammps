.. index:: fix pimd/mixed

fix pimd/mixed command
======================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID all pimd/mixed temp T keyword values ...

* ID is documented in :doc:`fix <fix>` command
* the group must be all
* temp T = positive temperature defining the ring springs
* zero or more keywords may be appended

  .. parsed-literal::

     *beads* values = group-ID count
       group-ID = static group assigned a bead count
       count = power of two from 1 to the number of partitions
     *sp* value = positive multiplier of Planck's constant
     *thermostat* values = pile tau seed
       tau = positive centroid relaxation time (time units)
       seed = positive random number seed

Examples
""""""""

.. code-block:: LAMMPS

   group quantum id 1 2
   fix pi all pimd/mixed temp 300 beads quantum 8 thermostat pile 100 19771

This example requires eight partitions. All other atoms have one independent
coordinate shared by the eight force-evaluation configurations. Their initial
coordinates and velocities must agree between partitions.

For three fixed groups with eight, two and one independent beads:

.. code-block:: LAMMPS

   group fine id 1
   group coarse id 2
   fix pi all pimd/mixed temp 300 beads fine 8 beads coarse 2 thermostat pile 100 19771

The two coarse beads are copied into partitions 1-4 and 5-8 respectively.
Unassigned atoms have one bead copied into all eight partitions. Groups must
not overlap. All copies of a particular independent bead must start with
the same unwrapped position and velocity.

Description
"""""""""""

.. versionadded:: TBD

Integrate fixed mixed-bead ring-polymer motion with optional local PILE
thermostatting. Without the *thermostat* keyword, the motion is deterministic
and conserves the fictitious Hamiltonian up to integration error. The specified
temperature defines the quantum springs and, when enabled, the PILE target.
The motion samples a mixed nuclear ensemble; it is not physical quantum
real-time dynamics.

With :math:`P` partitions and :math:`p_i` unique beads for atom :math:`i`, the
zero-based map is :math:`a_i(b)=\lfloor b p_i/P\rfloor`. Each partition evaluates
a complete physical configuration :math:`R_b=\{q_{i,a_i(b)}\}`. The physical
potential and spring energy are

.. math::

   U = \frac{1}{P}\sum_b E(R_b), \qquad
   U_s = \sum_i\frac{m_i p_i}{2\beta^2\hbar^2}
           \sum_a |q_{ia}-q_{i,a+1}|^2 .

Here :math:`\beta=1/(k_B T)`; the expressions include the corresponding mass,
length, time and energy conversion factors for the selected LAMMPS units.
The physical force on a unique coordinate is

.. math::

   F_{ia} = \frac{1}{P}\sum_{b:a_i(b)=a} f_i(R_b).

The fictitious mass is :math:`m_i/p_i`. Consequently the dynamical force stored
in the atom force array is :math:`p_i F_{ia}` and is used with mass :math:`m_i`.
It is the bucket average of the raw physical forces. Springs are propagated
analytically in orthonormal normal modes. They are not added to the atom force
array. Positions and velocities at output and restart boundaries are in
Cartesian bead coordinates. Only one representative of each unique bead is
integrated; copies are reconstructed for force evaluation.

For one classical environment atom, there is one independent position and
velocity. Evolving independent environments and averaging them afterward is
not this Hamiltonian. Reduced bead counts do not reduce the number of full
potential evaluations in this implementation.

The integrator applies a physical-force half kick, an exact free-ring half
propagation, the optional PILE step, another free-ring half propagation, and
the final physical-force half kick. With *thermostat pile*, the centroid mode
has friction :math:`1/\tau` and each internal mode has friction
:math:`2\omega_k`. Only independent mode representatives draw random numbers.
In consistent mechanical units their velocity variance is
:math:`p_i k_B T/m_i`; the implementation includes the unit conversion factor.
The classical environment therefore has one noise process per coordinate.

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The fix saves the bead assignment and spring settings in binary restart files.
Recreate it with the same options and groups. A changed assignment, spring
setting or thermostat setting is rejected. With PILE, the complete random
number state, including the cached Gaussian value, is saved for every rank.
PILE restarts require the same processor count in each partition.
Deterministic restarts may change processor counts or grids. Exact stochastic
continuation additionally requires unchanged atom ordering and decomposition;
changing sorting or processor layout does not preserve a random trajectory.
Specify the same :doc:`processors` grid before :doc:`read_restart` when exact
continuation is required; a binary restart does not force the original grid. Atom coordinates and bead velocities are stored by
:doc:`write_restart`.

The global scalar is the fictitious Hamiltonian :math:`K+U+U_s+B`. The seven-element
global vector contains, in order, :math:`U`, :math:`U_s`, fictitious kinetic
energy :math:`K`, the Hamiltonian including bias, the primitive kinetic estimator,
the centroid-virial kinetic estimator, and the total physical bias :math:`B`. All seven entries are extensive and use
energy units. These estimators are distinct from the conserved fictitious
Hamiltonian. Standard thermo temperature and potential energy retain their
single-partition meanings. The kinetic estimators use physical potential forces
before bias; their averages in a biased ensemble require appropriate reweighting.

The three per-atom columns report the number of unique beads, the one-based
unique bead index for this partition, and a representative flag (one or zero).
Copies with the same unique index are not independent statistical samples.

No :doc:`fix_modify` or :doc:`run` start/stop options apply. The fix is not
invoked during minimization.

Restrictions
""""""""""""

This fix belongs to the REPLICA package. It requires three dimensions,
consecutive atom IDs, an atom map, per-type masses, equal processor counts per
partition, identical cells and atom identities, and a fixed cell without
shrink-wrap boundaries. Unit styles and timesteps must match across partitions. Supported
unit styles are lj, real and metal. Bead counts and the partition count must
be powers of two; each atom may have any such count up to P, with
at least one atom using P beads. The all-classical limit uses one partition.
Bead groups must be static, nonoverlapping and unchanged during continuation.
Atom insertion/deletion and changing the cell are not supported.

Only :doc:`fix print <fix_print>` and :doc:`fix ave/time <fix_ave_time>` are
supported as companion fixes, along with one registered PIMD bias adapter
(see below). Constraints, additional integrators,
external thermostats, barostats, force modification, bosonic exchange and rRESPA are
not supported. Ring coordinates use the continuous unwrapped paths specified
by image flags. This method does not infer paths with periodic winding from
wrapped coordinates alone.

Bias coupling
"""""""""""""

:doc:`fix plumed <fix_plumed>` can apply *coordinate_mean*, *cv_mean*, or
*bias_mean* to this integrator. Define it after the integrator, with
*pimd_fix* naming this fix. The PLUMED input determines the collective
variable and the shared bias history; these modes are different functions:

.. math::

   B_c = B(s(\bar R)),\quad
   B_m = B\left(P^{-1}\sum_b s(R_b)\right),\quad
   B_d = P^{-1}\sum_b B(s(R_b)).

The centroid of each atom is its mean over its own unique beads. For a
physical per-copy gradient :math:`g_{ib}=-\partial B/\partial R_{ib}`, the
additional stored dynamical force is
:math:`p_i\sum_{b:a_i(b)=a} g_{ib}`. A classical environment atom receives
the complete summed feedback whenever the collective variable depends on it.
Do not add an independent bias history to each partition.

The scalar and fourth vector entry include the total bias exactly once,
regardless of the adapter's :doc:`fix_modify` energy setting. The first
vector entry remains the physical potential average. Restarting this fix
restores only its integration state; the bias adapter's state/history must
also be restored using that adapter's restart protocol. Path contraction
and mixed-bead pressure/barostats are not supported by this interface.

Related commands
""""""""""""""""

:doc:`fix pimd/nve <fix_pimd_nve>`, :doc:`fix pimd/langevin <fix_pimd>`

Default
"""""""

Unassigned atoms have one bead; sp = 1; thermostatting is off. Temperature has no default.
