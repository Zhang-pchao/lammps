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

#ifdef FIX_CLASS
// clang-format off
FixStyle(pimd/mixed,FixPIMDMixed);
// clang-format on
#else

#ifndef LMP_FIX_PIMD_MIXED_H
#define LMP_FIX_PIMD_MIXED_H

#include "fix_pimd_nve.h"
#include "pimd_mixed_map.h"

#include <array>
#include <map>
#include <memory>

namespace LAMMPS_NS {

class RanMars;

class FixPIMDMixed : public FixPIMDNVE {
 public:
  FixPIMDMixed(class LAMMPS *, int, char **);
  ~FixPIMDMixed() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void post_force(int) override;
  void end_of_step() override;
  double compute_scalar() override;
  double compute_vector(int) override;
  void *extract(const char *, int &) override;
  PIMDBiasCapabilities bias_capabilities() const override;
  void submit_bias(const PIMDBiasRegistration &, const PIMDBiasSubmission &, double *) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void restart(char *, int) override;

 private:
  std::vector<std::string> bead_groups;
  std::vector<int> group_beads, bead_counts, saved_counts;
  std::map<int, PIMDMixedMap> maps;
  std::array<double, 9> fixed_cell{};
  std::array<double, 7> values{};
  bool initialized = false;
  int metadata_capacity = 0;
  bigint unique_beads = 0;
  std::unique_ptr<RanMars> mixed_random;
  double thermostat_tau = 0.0;
  int thermostat_seed = 0;

  const PIMDMixedMap &atom_map(int) const;
  void initialize_counts();
  void validate_companions();
  void validate_copies();
  void unwrap_coordinates();
  void kick();
  void transform(double **, bool);
  void propagate_modes(double);
  void o_step() override;
  void replicate_velocities();
  void update_metadata();
};

}    // namespace LAMMPS_NS
#endif
#endif
