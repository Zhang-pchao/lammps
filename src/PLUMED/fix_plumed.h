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
FixStyle(plumed,FixPlumed);
// clang-format on
#else

#ifndef LMP_FIX_PLUMED_H
#define LMP_FIX_PLUMED_H

#include "fix.h"
#include "pimd_bias_interface.h"

#include <vector>

// forward declaration
namespace PLMD {
class Plumed;
}

namespace LAMMPS_NS {

class FixPlumed : public Fix {
 public:
  FixPlumed(class LAMMPS *, int, char **);
  ~FixPlumed() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  void *extract(const char *, int &) override;
  void reset_dt() override;
  int modify_param(int narg, char **arg) override;
  double memory_usage() override;

 private:
  enum {
    PATH_INTEGRAL_OFF,
    PATH_INTEGRAL_COORDINATE_MEAN,
    PATH_INTEGRAL_CV_MEAN,
    PATH_INTEGRAL_BIAS_MEAN
  };

  int path_contraction_flag;
  double path_contraction;
  std::vector<double> contracted_positions, contracted_forces, contraction_mean;

  PLMD::Plumed *p;                 // pointer to plumed object
  PIMDBiasInterface *pimd_core;    // typed capability, refreshed during init
  std::shared_ptr<PIMDBiasRegistration> pimd_registration;
  bigint last_pimd_update_step;    // history updates are distinct from force evaluations
  std::vector<double> bias_positions, bias_forces;
  int nlocal;                      // number of atoms local to this process
  int natoms;                      // total number of atoms
  int path_integral_mode;          // path-integral coupling mode
  int plumed_active;               // this partition runs a PLUMED instance
  int *gatindex;                   // array of atom indexes local to this process
  double *masses;                  // array of masses for local atoms
  double *charges;                 // array of charges for local atoms
  double *centroid_positions;      // local centroid coordinates passed to PLUMED
  double *centroid_forces;         // local centroid forces returned by PLUMED
  double *centroid_forces_all;     // tag-ordered centroid forces shared by all beads
  int nlevels_respa;               // this is something to enable respa
  double bias;                     // output bias potential
  class Compute *c_pe;             // Compute for the energy
  class Compute *c_press;          // Compute for the pressure
  int plumedNeedsEnergy;           // Flag to trigger calculation of the
                                   // energy and virial
  char *id_pe, *id_press;          // ID for potential energy and pressure compute
  char *nonfinite_trace_prefix;    // opt-in diagnostic output prefix
  char *id_pimd;                   // ID for the coupled PIMD integrator

  void check_path_integral_compatibility();
  void update_atom_data();
  void post_force_beads(const PIMDBiasCoordinates &);
  void post_force_centroid(const PIMDBiasCoordinates &);
  void prepare_contracted_coordinates(const PIMDBiasCoordinates &);
  void pullback_contracted_forces();
  void trace_nonfinite_state(const char *, const double *, bool, const double * = nullptr,
                             double = 1.0);
  void trace_centroid_nonfinite(const char *);
};

};    // namespace LAMMPS_NS

#endif
#endif
