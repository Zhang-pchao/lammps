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

#ifdef FIX_CLASS
// clang-format off
FixStyle(pimd/adaptive,FixPIMDAdaptive);
// clang-format on
#else

#ifndef LMP_FIX_PIMD_ADAPTIVE_H
#define LMP_FIX_PIMD_ADAPTIVE_H

#include "fix_pimd_nve.h"
#include "pimd_mixed_map.h"
#include "pimd_pes_view.h"

#include <array>
#include <memory>

namespace LAMMPS_NS {
class RanMars;

class FixPIMDAdaptive : public FixPIMDNVE {
 public:
  FixPIMDAdaptive(class LAMMPS *, int, char **);
  ~FixPIMDAdaptive() override;
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
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int maxsize_restart() override;
  int size_restart(int) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void restart(char *, int) override;

 private:
  // Published columns: y(3), auxiliary velocity(3), conjugate force(3),
  // amplitude(1), amplitude gradient(3). All use the same per-atom callbacks.
  static constexpr int NCOLS = 13;
  PIMDPESView pes_view;
  void refresh_pes_view();
  void capture_pes_snapshot();
  bool pes_snapshot_matches();
  bool pes_snapshot_enabled = false, pes_snapshot_valid = false;
  // Canonical tag order, independent of local sorting and shared bead scratch.
  std::vector<double> pes_snapshot_auxiliary, pes_snapshot_centroid;
  std::vector<unsigned char> pes_snapshot_seen;
  std::array<double, 2> pes_snapshot_counts{};
  bool initialized = false, restored = false;
  int capacity = 0;
  std::array<double, 3> center{}, fixed_center{}, reference_velocity{};
  int region_axis = -1, anchor_bit = 0, quantum_bit = 0;
  double anchor_mass = 0.0;
  std::string anchor_name, quantum_name;
  std::vector<int> anchor_members, quantum_members;
  void validate_groups();
  void update_anchor(const std::vector<double> &, bool);
  double core = 0.0, outer = 0.0, thermostat_tau = 0.0;
  int thermostat_seed = 0;
  std::array<double, 9> fixed_cell{};
  std::array<double, 5> values{};
  std::unique_ptr<PIMDMixedMap> modes;
  std::vector<double> normal_mode_forward, normal_mode_inverse;
  double normal_mode_eigenvalue = 0.0;
  std::unique_ptr<RanMars> random;
  std::vector<double *> rows;
  std::vector<double> offsets;
  std::vector<double> collective_send, collective_values;
  bool force_summary_enabled = false;
  std::vector<double> force_summary;
  std::array<double, 2> force_summary_counts{};
  // Scratch is refilled on every use; it never caches coordinates across migration.
  std::vector<double> centroids, mean_velocity, bias_centers, bias_forces;
  std::vector<double *> bias_force_rows;
  void validate_companions();
  bool profile_enabled = false;
  std::array<double, 24> phase_profile{};
  void inter_replica_comm(double **);
  void collect(int);
  void transform(int, bool);
  void map_positions();
  void map_velocities(bool positions_just_mapped = false);
  void kick();
  void propagate(double);
  void o_step() override;
  void pullback();
};
}    // namespace LAMMPS_NS
#endif
#endif
