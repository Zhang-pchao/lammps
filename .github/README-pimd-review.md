# Personal-fork PIMD validation

`fork-pimd-review.yml` is an internal review workflow. Exclude it from a later
official contribution.

- `native` runs the MPI PIMD, RNG and restart groups in SMALLBIG and BIGBIG.
- `coupling` builds a pinned public PLUMED fork commit and runs all four native
  PLUMED coupling groups, including volume/virial, normalization and odd/even
  bead contraction tests. The pin supplies pending shared-path support and is
  not a claim that all these PLUMED features have been merged upstream.
- `kokkos` runs four native dump cases in both default mode and explicitly
  enabled KOKKOS Serial mode. Each required CTest group is invoked separately
  with `--no-tests=error` so missing registrations cannot pass silently.

The hosted KOKKOS job uses CPU shared memory. Changes involving separate host
and device copies also require an explicitly enabled GPU run of the same
native regression. The CPU job alone cannot detect every missing device-to-host
synchronization. Preserve the failing-before/passing-after GPU result and the
source identity used for that check.

The box-lifetime regression is part of `PlumedPIMDMPI`. The host-output regression
is `DumpCustomTest.rerun_forces`; it reads the actual dump rather than using an
atom-extraction call that could synchronize data and hide the defect.
