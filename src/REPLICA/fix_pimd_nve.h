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
FixStyle(pimd/nve,FixPIMDNVE);
// clang-format on
#else

#ifndef LMP_FIX_PIMD_NVE_H
#define LMP_FIX_PIMD_NVE_H

#include "fix.h"

#include <vector>

namespace LAMMPS_NS {

class FixPIMDNVE : public Fix {
 public:
  enum { PIMD, NMPIMD, CMD };
  enum { PHYSICAL, NORMAL };
  enum { BAOAB, OBABO };
  enum { SINGLE_PROC, MULTI_PROC };

  FixPIMDNVE(class LAMMPS *, int, char **);
  ~FixPIMDNVE() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void end_of_step() override;
  double compute_vector(int) override;

 protected:
  // Deferred construction initializes storage without parsing options or creating computes.
  FixPIMDNVE(class LAMMPS *, int, char **, bool);

  int method, fmmode, integrator, np;
  double inverse_np, normal_mode_centroid_force_scale;
  double temp, hbar, kt, beta, beta_np;
  int mapflag, removecomflag;
  double masstotal;
  double omega_np, fbond, spring_energy, sp;
  double fmass, *mass;

  MPI_Comm rootworld;
  int me, nprocs, ireplica, nreplica, nprocs_universe;
  int ntotal, maxlocal, x_last, x_next, cmode, sizeplan, maxsend;
  int *plansend, *planrecv;
  tagint *tagsend, *tagrecv;
  double *bufsend, *bufrecv, **bufbeads;
  double **bufsorted, **bufsortedall;
  int *counts, *displacements;

  double *lam, **M_x2xp, **M_xp2x, **M_f2fp, **M_fp2f;
  int *modeindex;
  int defer_normal_mode_force, normal_mode_force_pending;
  int bead_bias_virial_pending, centroid_bias_virial_pending;
  double dtv, dtf, dtv2, dtv3;
  double *_omega_k, *Lan_s, *Lan_c;

  double **xc, *xcall, **x_unwrap;
  int maxxc, maxunwrap;
  double vir, vir_, centroid_vir;
  double t_prim, t_vir, t_cv, p_prim, p_vir, p_cv, p_md;
  double pote, tote, totke, ke_bead, se_bead, pe_bead, total_spring_energy;
  char *id_pe, *id_press;
  class Compute *c_pe, *c_press;

  void validate_partition_topology(bool);
  void check_atom_consistency();
  void initialize_partition_communication(bool = true);
  void create_energy_computes();
  void comm_init(bool = false);
  void inter_replica_comm(double **);
  void ring_collect(const std::vector<tagint> &, double **, std::vector<double> &);
  void reallocate();
  void reallocate_x_unwrap();
  void reallocate_xc();
  void nmpimd_init();
  void init_normal_mode_coefficients();
  void nmpimd_transform(double **, double **, double *);
  double **normal_mode_transform_buffer();
  void prepare_normal_mode_forces();
  void collect_xc();
  void remove_com_motion();

  virtual void prepare_coordinates();
  virtual void b_step();
  virtual void qc_step();
  virtual void a_step();
  virtual void o_step();
  void q_step();
  virtual void spring_force();
  virtual void setup_subclass_state();
  virtual void after_force_transform_hook();
  virtual void trace_nonfinite_state(const char *, const char *);

  void compute_totke();
  virtual void compute_spring_energy();
  void compute_pote();
  void compute_tote();
  virtual void compute_t_prim();
  void compute_t_vir();
  void compute_p_prim();
  void compute_p_cv();
  void compute_vir();
  void compute_xf_vir();
  virtual void compute_cvir();
};

}    // namespace LAMMPS_NS

#endif
#endif
