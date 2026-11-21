.. index:: fix plumed

fix plumed command
==================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID plumed keyword value ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* plumed = style name of this fix command
* keyword = *plumedfile* or *outfile* or *path_integral* or *pimd_fix* or *path_contraction*

  .. parsed-literal::

       *plumedfile* arg = name of PLUMED input file to use (default: NULL)
       *outfile* arg = name of file on which to write the PLUMED log (default: NULL)
       *path_integral* arg = *off* or *coordinate_mean* or *cv_mean* (default: *off*)
       *pimd_fix* arg = ID of the path-integral integrator
       *path_contraction* arg = fixed coordinate contraction between 0 and 1

Examples
""""""""

.. code-block:: LAMMPS

   fix bias all plumed plumedfile plumed.dat outfile p.log

Description
"""""""""""

This fix instructs LAMMPS to call the `PLUMED <plumedhome_>`_ library, which
allows one to perform various forms of trajectory analysis on the fly
and to also use methods such as umbrella sampling and metadynamics to
enhance the sampling of phase space.

The documentation included here only describes the fix plumed command
itself.  This command is LAMMPS specific, whereas most of the
functionality implemented in PLUMED will work with a range of MD codes,
and when PLUMED is used as a stand alone code for analysis.  The full
`documentation for PLUMED <plumeddocs_>`_ is available online and included
in the PLUMED source code.  The PLUMED library development is hosted at
`https://github.com/plumed/plumed2 <https://github.com/plumed/plumed2>`_
A detailed discussion of the code can be found in :ref:`(Tribello) <Tribello>`.

There is an example input for using this package with LAMMPS in the
examples/PACKAGES/plumed directory.

----------

The command to make LAMMPS call PLUMED during a run requires two keyword
value pairs pointing to the PLUMED input file and an output file for the
PLUMED log. The user must specify these arguments every time PLUMED is
to be used.  Furthermore, the fix plumed command should appear in the
LAMMPS input file **after** relevant input parameters (e.g. the timestep)
have been set.

The *group-ID* entry is ignored. LAMMPS will always pass all the atoms
to PLUMED and there can only be one instance of the plumed fix at a
time. The way the plumed fix is implemented ensures that the minimum
amount of information required is communicated.  Furthermore, PLUMED
supports multiple, completely independent collective variables, multiple
independent biases and multiple independent forms of analysis.  There is
thus really no restriction in functionality by only allowing only one
plumed fix in the LAMMPS input.

The *plumedfile* keyword allows the user to specify the name of the
PLUMED input file.  Instructions as to what should be included in a
plumed input file can be found in the `documentation for PLUMED
<plumeddocs_>`_

The *outfile* keyword allows the user to specify the name of a file in
which to output the PLUMED log.  This log file normally just repeats the
information that is contained in the input file to confirm it was
correctly read and parsed.  The names of the files in which the results
are stored from the various analysis options performed by PLUMED will
be specified by the user in the PLUMED input file.

Path-integral coupling
----------------------

.. versionadded:: TBD

The *path_integral* keyword couples a whole ring polymer to one bias potential.
Define :doc:`fix pimd/langevin <fix_pimd>` first, then define fix plumed after
all other fixes that modify forces.  The *pimd_fix* keyword identifies the
integrator by its fix ID.  The integrator must act on group *all*.  Each bead
partition must contain the same atoms, masses, types, and simulation cell and
use the same number of MPI ranks.  Atom ownership within each partition may
change.  With *path_integral off*, ordinary classical PLUMED behavior is retained.

For example, launch LAMMPS with ``-partition 4x1`` and use:

.. code-block:: LAMMPS

   fix dynamics all pimd/langevin method nmpimd ensemble nvt &
       integrator obabo fmmode physical fmass 1.0 sp 1.0 temp 300 &
       thermostat PILE_L 2468 tau 0.1 fixcom no
   fix bias all plumed plumedfile plumed.dat outfile plumed.log &
       path_integral cv_mean pimd_fix dynamics path_contraction 0.5
   fix_modify bias energy yes

Here ``dynamics`` and ``bias`` are user-selected IDs, not keywords or style
names.  To use constant pressure with normal-mode PIMD, select ``ensemble npt``
and supply the pressure options documented for :doc:`fix pimd/langevin <fix_pimd>`.
Both bias modes support *method pimd* with *ensemble nvt* and *method nmpimd*
with *ensemble nvt*, *nph*, or *npt*.  Pressure coupling uses *barostat BZP*
and supports orthogonal cells with isotropic or diagonal pressure control.
The existing integrator owns cell propagation and the pressure estimator;
PLUMED supplies the bias forces and virial before the normal-mode force
transformation.  Contraction remains fixed as the cell evolves.  Other fixes
that change the simulation cell cannot be combined with this coupling.

With *coordinate_mean*, PLUMED evaluates the CVs on the Cartesian centroid of
all beads.  Only the first partition evaluates PLUMED, and its forces are
shared across the path.  Do not use PLUMED ``ENSEMBLE`` for this mode.

With *cv_mean*, each partition evaluates the same PLUMED input on its bead
coordinates.  Use PLUMED ``ENSEMBLE`` to average the CVs before applying the
bias, for example:

.. code-block:: text

   d: DISTANCE ATOMS=1,2
   mean: ENSEMBLE ARG=d
   restraint: RESTRAINT ARG=mean.d AT=0.3 KAPPA=100

This example uses PLUMED's default units.  PLUMED converts the LAMMPS units
through its engine interface.  All partitions must use the same bias graph;
independent per-bead bias histories do not define the coupled whole-path bias.
The *cv_mean* mode requires multiple partitions and PLUMED with ``ENSEMBLE``
support.

The optional *path_contraction* keyword applies only to *cv_mean*.  Let
:math:`P` be the number of beads and :math:`\lambda` the fixed contraction.
For bead coordinates :math:`\mathbf r_b` and centroid
:math:`\bar{\mathbf r}`, the coordinates passed to PLUMED are

.. math::

   \widetilde{\mathbf r}_b = \bar{\mathbf r}
      + \lambda(\mathbf r_b-\bar{\mathbf r}),\qquad
   s_\lambda = \frac{1}{P}\sum_b s(\widetilde{\mathbf r}_b),\qquad
   V_\lambda=V(s_\lambda).

Thus zero contraction evaluates a centroid CV, while unit contraction
evaluates the bead-mean CV.  Intermediate values contract coordinates before
evaluating the nonlinear CV; they do not interpolate the two endpoint CVs.
The adapter applies the full chain rule to the bead forces.  The integrator
applies its factor :math:`P` exactly once to the physical bias forces and
virial.  The scalar reported by fix plumed is the physical whole-path bias on
the first partition and zero on the other partitions, so summing it over
partitions counts the bias once.  The integrator's energy uses its documented
scaled-temperature convention.

Explicit contraction uses unwrapped bead coordinates and the current cell.
Omitting *path_contraction* retains the original wrapped-coordinate interface.
Consequently, explicit ``path_contraction 1`` and an omitted contraction agree
only when the CV's periodic-image convention makes their coordinates
equivalent.  Contraction does not remove beads or change the coordinates used
for physical quantum estimators.  Its value must be identical in all partitions
and remain fixed during the trajectory.

Path-integral modes require Verlet integration and do not support minimization,
energy-dependent PLUMED actions, or changing the atom count.  Bias forces must
be supplied for every force evaluation, including ``run 0``.  Repeated force
evaluations at the same timestep do not deposit additional bias history.
LAMMPS restart files do not contain PLUMED history; use PLUMED's own restart
options when continuing an adaptive bias.  Statistical restart compatibility
does not imply identical stochastic trajectories after changes in atom order
or MPI layout.

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When performing a restart of a calculation that involves PLUMED you
must include a RESTART command in the PLUMED input file as detailed in
the `PLUMED documentation <plumeddocs_>`_.  When the restart command
is found in the PLUMED input PLUMED will append to the files that were
generated in the run that was performed previously.  No part of the
PLUMED restart data is included in the LAMMPS restart files.
Furthermore, any history dependent bias potentials that were
accumulated in previous calculations will be read in when the RESTART
command is included in the PLUMED input.

The :doc:`fix_modify <fix_modify>` *energy* option is supported by
this fix to add the energy change from the biasing force added by
PLUMED to the global potential energy of the system as part of
:doc:`thermodynamic output <thermo_style>`.  The default setting for
this fix is :doc:`fix_modify energy yes <fix_modify>`.

The :doc:`fix_modify <fix_modify>` *virial* option is supported by
this fix to add the contribution from the biasing force to the global
pressure of the system via the :doc:`compute pressure
<compute_pressure>` command.  This can be accessed by
:doc:`thermodynamic output <thermo_style>`.  The default setting for
this fix is :doc:`fix_modify virial yes <fix_modify>`.

This fix computes a global scalar which can be accessed by various
:doc:`output commands <Howto_output>`.  The scalar is the PLUMED
energy mentioned above.  The scalar value calculated by this fix is
"extensive".

Note that other quantities of interest can be output by commands that
are native to PLUMED.

Restrictions
""""""""""""

This fix is part of the PLUMED package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

There can only be one fix plumed command active at a time.

Related commands
""""""""""""""""

:doc:`fix smd <fix_smd>`
:doc:`fix colvars <fix_colvars>`

Default
"""""""

The default options are plumedfile = NULL, outfile = NULL, and path_integral = off.
No path_contraction is applied unless it is specified.

----------

.. _Tribello:

**(Tribello)** G.A. Tribello, M. Bonomi, D. Branduardi, C. Camilloni and G. Bussi, Comp. Phys. Comm 185, 604 (2014)

.. _plumeddocs: https://www.plumed.org/doc.html

.. _plumedhome: https://www.plumed.org/
