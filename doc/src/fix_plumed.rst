.. index:: fix plumed

fix plumed command
==================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID plumed keyword value ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* plumed = style name of this fix command
* keyword = *plumedfile* or *outfile* or *path_integral* or *pimd_fix*

  .. parsed-literal::

       *plumedfile* arg = name of PLUMED input file to use (default: NULL)
       *outfile* arg = name of file on which to write the PLUMED log (default: NULL)
       *path_integral* arg = *off*, *centroid*, or *bead_mean* (default: off)
       *pimd_fix* arg = ID of the coupled fix pimd/langevin

Examples
""""""""

.. code-block:: LAMMPS

   fix pl all plumed plumedfile plumed.dat outfile p.log
   fix pl all plumed plumedfile plumed.dat outfile p.log path_integral centroid pimd_fix fpimd
   fix pl all plumed plumedfile plumed.dat outfile p.log path_integral bead_mean pimd_fix fpimd

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

.. versionadded:: TBD

The *path_integral centroid* setting couples PLUMED to the Cartesian coordinate
centroid provided by the :doc:`fix pimd/langevin <fix_pimd>` command selected
with *pimd_fix*.  The PIMD fix must be defined before fix plumed.  One PLUMED
state is created on partition zero, so there is one bias history rather than an
independent history for each bead.  If the centroid bias force is
:math:`\mathbf{F}_c`, this fix adds :math:`\mathbf{F}_c/P` to every one of the
:math:`P` Cartesian beads.

This mode biases a collective variable evaluated from the coordinate centroid,
which is generally different from averaging the collective variable over the
beads.  The scalar bias energy and bias virial are nonzero only on partition
zero so they are counted once in the ring-polymer Hamiltonian.  Bead-resolved
trajectories are still required to reconstruct a bead-defined quantum free
energy.

The *path_integral bead_mean* setting creates one PLUMED instance on every
bead partition and enables PLUMED's multiple-replica communication.  Use the
PLUMED ``ENSEMBLE`` action to define the arithmetic bead mean of a
collective variable and apply biases only to that mean.  For example:

.. code-block:: text

   d: DISTANCE ATOMS=1,2
   mean: ENSEMBLE ARG=d
   bias: RESTRAINT ARG=mean.d AT=0.5 KAPPA=10

If :math:`S=P^{-1}\sum_b s(\mathbf{R}_b)`, PLUMED propagates the bias force
with the chain-rule factor :math:`1/P`, so bead :math:`b` receives
:math:`-(\partial V/\partial S)\nabla_b s/P`.  Applying a bias directly to
``d`` instead of ``mean.d`` creates independent per-bead biases and is not a
bead-mean calculation.

All PLUMED instances evaluate the same bead-mean bias in lockstep.  PLUMED
adds the partition suffix to its output and restart files.  The LAMMPS scalar
bias energy is reported only on partition zero so that it is counted once;
the chain-rule force and virial contributions remain local to every bead.
Do not use a multiple-walker option to combine the PIMD beads: they are parts
of one ring polymer, not statistically independent walkers.

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

Both path-integral modes require :doc:`fix pimd/langevin <fix_pimd>` with
*method pimd* and *ensemble nvt*.  The *centroid* mode additionally requires
one MPI rank per bead, a fixed atom count, consecutive atom IDs, and an atom
map.  The *bead_mean* mode requires multiple LAMMPS partitions and a PLUMED
input that explicitly routes each bias through ``ENSEMBLE``.  Neither
mode supports energy-dependent PLUMED actions, minimization, or r-RESPA.
The default *path_integral off* setting remains incompatible with path-integral
fixes.

Related commands
""""""""""""""""

:doc:`fix smd <fix_smd>`
:doc:`fix colvars <fix_colvars>`

Default
"""""""

The default options are plumedfile = NULL, outfile = NULL, and path_integral = off.

----------

.. _Tribello:

**(Tribello)** G.A. Tribello, M. Bonomi, D. Branduardi, C. Camilloni and G. Bussi, Comp. Phys. Comm 185, 604 (2014)

.. _plumeddocs: https://www.plumed.org/doc.html

.. _plumedhome: https://www.plumed.org/
