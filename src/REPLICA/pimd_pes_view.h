/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_PIMD_PES_VIEW_H
#define LMP_PIMD_PES_VIEW_H

#include "lmptype.h"

#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

// Read-only input to an optional PES evaluator. This is not a force-submission
// or bias interface. Every partition/rank calls extract collectively, after
// fix initialization. Data is refreshed on every call, including at one step.
// The caller must copy values before another extract and must preserve tags
// when building cross-partition or ghost mappings. No timestep cache is valid.
struct PIMDPESView {
  static constexpr int VERSION = 1;
  int version = VERSION;
  int beads = 0, bead = 0, nlocal = 0;
  bigint natoms = 0;
  std::uint64_t generation = 0;
  std::vector<tagint> tags;
  std::vector<double> centroid;
  // Nonzero amplitude implies derivative support, even for zero displacement.
  std::vector<double> amplitude;
};

}    // namespace LAMMPS_NS
#endif
