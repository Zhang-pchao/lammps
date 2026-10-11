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

#ifndef LMP_PIMD_BIAS_INTERFACE_H
#define LMP_PIMD_BIAS_INTERFACE_H

#include "lmptype.h"

#include <cstdint>
#include <memory>

namespace LAMMPS_NS {

class Fix;
class PIMDBiasInterface;

enum class PIMDBiasMode { COORDINATE_MEAN, CV_MEAN, BIAS_MEAN };

// The token outlives either Fix. The core clears owner before destruction;
// an adapter can then release its registration without dereferencing a dead Fix.
struct PIMDBiasRegistration {
  PIMDBiasInterface *owner = nullptr;
  Fix *adapter = nullptr;
  PIMDBiasMode mode = PIMDBiasMode::COORDINATE_MEAN;
  bool contraction = false;
};

struct PIMDBiasCapabilities {
  int beads;
  bool coordinate_mean;
  bool bead_modes;
  bool fixed_cell_contraction;
};

// The core validates matching cells, masses, types and group membership before
// exposing these views. Adapters use the current LAMMPS cell; they do not change
// coordinates or ownership between obtaining a view and submitting its forces.
// Borrowed, read-only views in local owned-atom order, valid until submit() or
// the next force evaluation. centroid_by_tag uses 3*(tag-1)+component. A new
// evaluation has a new generation even for repeated run 0 at the same timestep.
struct PIMDBiasCoordinates {
  std::uint64_t generation;
  bigint timestep;
  int nlocal;
  const tagint *tags;
  const double *const *beads;              // wrapped Cartesian coordinates
  const double *const *unwrapped_beads;    // coherent lift used for contraction
  const double *centroid_by_tag;
};

// force is -grad_b(B), separate from atom->f, in the view's owned-atom order.
// energy is the physical B share for this partition, replicated on its ranks.
// virial is the local physical bias virial share, -dB/d(affine strain),
// in the native LAMMPS order (xx, yy, zz, xy, xz, yz).
// Physical kinetic estimators are captured by the core before subsequent
// post-force fixes. This submission updates biased pressure without redefining
// those estimators to include arbitrary external force modifiers.
// The core applies the dynamical factor P exactly once and completes force
// processing before returning. No caller may transform atom->f itself.
struct PIMDBiasSubmission {
  std::uint64_t generation;
  int nlocal;
  const double *force;
  double energy;
  const double *virial;
};

class PIMDBiasInterface {
 public:
  virtual ~PIMDBiasInterface() noexcept(false) = default;
  virtual PIMDBiasCapabilities bias_capabilities() const = 0;
  // Registration and submission are collective over all bead partitions.
  virtual std::shared_ptr<PIMDBiasRegistration> register_bias(Fix *, PIMDBiasMode, bool) = 0;
  virtual void unregister_bias(PIMDBiasRegistration &) = 0;
  // Coordinate acquisition is also collective, including ranks with no atoms.
  virtual PIMDBiasCoordinates bias_coordinates(const PIMDBiasRegistration &) const = 0;
  // dynamical_virial is the adapter's Fix::virial output. It is filled before
  // core pressure/energy refresh, so native thermo sees the same current tally.
  virtual void submit_bias(const PIMDBiasRegistration &, const PIMDBiasSubmission &,
                           double *dynamical_virial) = 0;
};

}    // namespace LAMMPS_NS

#endif
