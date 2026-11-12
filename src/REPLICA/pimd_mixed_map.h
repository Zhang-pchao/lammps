/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_PIMD_MIXED_MAP_H
#define LMP_PIMD_MIXED_MAP_H

#include "math_const.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace LAMMPS_NS {

// Fixed nested time slices for one atom.  Partition copies are force-evaluation
// coordinates; only the representative of each bead is an independent state.
class PIMDMixedMap {
 public:
  PIMDMixedMap(int partitions, int beads) : partitions_(partitions), beads_(beads), stride_(0)
  {
    if (!valid(partitions, beads))
      throw std::invalid_argument("Mixed bead counts must be nested positive powers of two");
    stride_ = partitions / beads;
  }

  static bool valid(int partitions, int beads)
  {
    return (partitions > 0) && (beads > 0) && (beads <= partitions) &&
        ((partitions & (partitions - 1)) == 0) && ((beads & (beads - 1)) == 0);
  }

  int partitions() const { return partitions_; }
  int beads() const { return beads_; }
  int copies_per_bead() const { return stride_; }

  int bead_index(int partition) const
  {
    if ((partition < 0) || (partition >= partitions_))
      throw std::out_of_range("Mixed bead partition index is out of range");
    return partition / stride_;
  }

  int representative(int bead) const
  {
    check_bead(bead);
    return bead * stride_;
  }

  bool is_representative(int partition) const
  { return representative(bead_index(partition)) == partition; }

  // Return the physical -dU/dq, where U=sum_b E(R_b)/P.  The caller supplies
  // one force component per expanded configuration, separated by slice_stride.
  // Conversion to the m_i/p_i fictitious mass is a separate integration step.
  double physical_force(const double *force, std::ptrdiff_t slice_stride, int bead) const
  {
    if (!force || (slice_stride <= 0))
      throw std::invalid_argument("Mixed bead forces require nonempty strided storage");
    const int begin = representative(bead);
    double sum = 0.0;
    for (int b = begin; b < begin + stride_; ++b) sum += force[b * slice_stride];
    return sum / partitions_;
  }

  // Orthonormal real Fourier convention: centroid, cosine modes, Nyquist,
  // then sine modes.  The inverse transform is the transpose.
  double coefficient(int mode, int bead) const
  {
    check_bead(mode);
    check_bead(bead);
    const double norm = 1.0 / std::sqrt(static_cast<double>(beads_));
    if (mode == 0) return norm;
    if (mode == beads_ / 2) return (bead % 2 == 0) ? norm : -norm;
    const double phase = MathConst::MY_2PI * mode * static_cast<double>(bead) / beads_;
    if (mode < beads_ / 2) return MathConst::MY_SQRT2 * norm * std::cos(phase);
    return MathConst::MY_SQRT2 * norm * std::sin(phase);
  }

  double eigenvalue(int mode) const
  {
    check_bead(mode);
    const double sine = std::sin(MathConst::MY_PI * mode / beads_);
    return 4.0 * sine * sine;
  }

 private:
  int partitions_, beads_, stride_;

  void check_bead(int bead) const
  {
    if ((bead < 0) || (bead >= beads_))
      throw std::out_of_range("Mixed bead or mode index is out of range");
  }
};

}    // namespace LAMMPS_NS

#endif
