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
FixStyle(pimd/langevin,FixPIMDLangevin);
// clang-format on
#else

#ifndef FIX_PIMD_LANGEVIN_H
#define FIX_PIMD_LANGEVIN_H

#include "fix_pimd_nve.h"

namespace LAMMPS_NS {

class FixPIMDLangevin : public FixPIMDNVE {
 public:
  enum { ISO, ANISO, TRICLINIC };
  enum { PILE_L };
  enum { MTTK, BZP };
  enum { NVE, NVT, NPH, NPT };

  FixPIMDLangevin(class LAMMPS *, int, char **);
  ~FixPIMDLangevin() override;
  void end_of_step() override;
  void write_restart(FILE *) override;
  void restart(char *, int) override;
  double compute_vector(int) override;
  void *extract(const char *, int &) override;

 protected:
  FixPIMDLangevin(class LAMMPS *, int, char **, bool);
  int thermostat, barostat, ensemble;
  double fixedpoint[3];

  char *nonfinite_trace_prefix;
  int nonfinite_trace_nmax;
  double **nonfinite_trace_last_x, **nonfinite_trace_last_v, **nonfinite_trace_last_f;
  tagint *nonfinite_trace_last_tag;
  char nonfinite_trace_last_stage[96];
  void trace_nonfinite_state(const char *, const char *) override;

  double gamma, c1, c2, tau;
  double *tau_k, *c1_k, *c2_k;
  double pilescale, Lan_temp;
  class RanMars *random;
  int tstat_flag;
  void langevin_init();
  void setup_subclass_state() override;
  void b_step() override;
  void qc_step() override;
  void o_step() override;

  int pstat_flag, pstyle;
  double W, tau_p, Pext, p_hydro, totenthalpy, Vcoeff;
  int pdim;
  int p_flag[6];
  double p_target[6], vw[6], ke_tensor[6], c_vir_tensor[6], stress_tensor[6];
  double vol0 = 0.0;
  void baro_init();
  void press_v_step();
  void press_o_step();
  void compute_stress_tensor();
  void compute_cvir() override;
  void compute_totenthalpy();

  int size_restart_global();
  int pack_restart_data(double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
