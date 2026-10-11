.. index:: fix pimd/nve

fix pimd/nve command
================================================================================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID pimd/nve keyword value ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* zero or more keyword/value pairs may be appended

  .. parsed-literal::

     *method* value = *pimd* or *nmpimd*
     *integrator* value = *obabo* or *baoab*
     *fmmode* value = *physical* or *normal*
     *fmass* value = positive fictitious mass scaling factor, at most the bead count
     *temp* value = positive temperature that defines the ring-polymer springs
     *sp* value = positive scaling factor on Planck's constant

Examples
""""""""

.. code-block:: LAMMPS

   fix integrator all pimd/nve method nmpimd temp 300.0
   fix integrator all pimd/nve method pimd fmmode physical temp 300.0

Description
"""""""""""

.. versionadded:: TBD

Integrate ring-polymer motion without a thermostat or barostat. The
temperature sets the spring frequency; it does not impose a kinetic
temperature. The ring-polymer Hamiltonian, normal-mode normalization,
fictitious masses, coordinate conventions, and output quantities match
:doc:`fix pimd/langevin ensemble nve <fix_pimd>`. The *nmpimd* method
propagates the free normal modes analytically; the *pimd* method uses
Cartesian bead forces including the ring springs.

The keyword meanings are the same as for *pimd/langevin*. Without a
thermostat or barostat, *obabo* and *baoab* give the same deterministic
splitting. This command does not change the input syntax or the
Nose-Hoover chain convention of *pimd/nvt*.

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix has no thermostat, barostat, or random-number state to save.
Atom coordinates and velocities are stored by :doc:`write_restart`.
Recreate the fix with the same bead count and options when continuing a
restart.

The fix computes the same 10-element global vector as *pimd/langevin*
with *ensemble nve*, with the same bead-local and global reduction
conventions. The first seven entries are extensive; the three pressure
entries are intensive. See :doc:`fix pimd/langevin <fix_pimd>` for the
entry definitions.

No :doc:`fix_modify` options or :doc:`run` *start/stop* options apply.
This fix is not invoked during energy minimization.

Restrictions
""""""""""""

This fix is part of the REPLICA package. The simulation must be
three-dimensional, with an atom map, consecutive atom IDs, and per-type
masses. Bead partitions must have matching cells, atom types, masses,
group membership, and numbers of MPI processes. See :doc:`fix pimd/langevin
<fix_pimd>` for partition setup and the interpretation of bead output.

The *normal* fictitious mass mode requires *method nmpimd*. Coupling this
style to an external bias is not enabled. Use the supported combinations
documented for :doc:`fix plumed <fix_plumed>` when a PLUMED bias is required.

Related commands
""""""""""""""""

:doc:`fix pimd/langevin <fix_pimd>`, :doc:`fix pimd/nvt <fix_pimd>`

Default
"""""""

The defaults are method = nmpimd, integrator = obabo, fmmode = physical,
fmass = 1.0, temp = 298.15, and sp = 1.0.
