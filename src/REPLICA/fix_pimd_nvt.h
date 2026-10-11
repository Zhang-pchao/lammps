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
FixStyle(pimd,FixPIMDNVT);
FixStyle(pimd/nvt,FixPIMDNVT);
// clang-format on
#else

#ifndef FIX_PIMD_NVT_H
#define FIX_PIMD_NVT_H

#include "fix_pimd_nve.h"

namespace LAMMPS_NS {

class FixPIMDNVT : public FixPIMDNVE {
 public:
  FixPIMDNVT(class LAMMPS *, int, char **);
  ~FixPIMDNVT() override;

  int setmask() override;

  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int maxsize_restart() override;
  int size_restart(int) override;
  double compute_vector(int) override;

  void *extract(const char *, int &) override;

 protected:
  // Legacy NVT uses its original force/coordinate normalization and per-atom NHC.
  double virial;
  void prepare_coordinates() override;
  virtual void pre_spring_force_estimators();
  void spring_force() override;
  void vir_estimator();
  void init_nvt_normal_modes();

  /* Nose-hoover chain integration */

  int nhc_offset_one_1, nhc_offset_one_2;
  int nhc_size_one_1, nhc_size_one_2;
  int nhc_nchain;
  bool nhc_ready;
  double nhc_temp, t_sys;

  double **nhc_eta;        /* coordinates of NH chains for ring-polymer beads */
  double **nhc_eta_dot;    /* velocities of NH chains                         */
  double **nhc_eta_dotdot; /* acceleration of NH chains                       */
  double **nhc_eta_mass;   /* mass of NH chains                               */

  void nhc_init();
  void nhc_update_v();
  void nhc_update_x();
};

}    // namespace LAMMPS_NS

#endif
#endif
