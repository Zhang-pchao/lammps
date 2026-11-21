/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Package      FixPIMDLangevin
   Purpose      Path Integral Molecular Dynamics with Langevin Thermostat

   Yifan Li @ Princeton University (yifanl0716@gmail.com)
   Current Features:
   - Multi-processor parallelism for each bead
   - White-noise Langevin thermostat
   - Bussi-Zykova-Parrinello barostat (isotropic and anisotropic)
   - Several quantum estimators
   Futher plans:
   - Triclinic barostat
------------------------------------------------------------------------- */

#include "fix_pimd_langevin.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "universe.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;
using MathConst::MY_PI;
using MathConst::MY_SQRT2;
using MathConst::THIRD;
using MathSpecial::powint;

namespace {
std::map<int, std::string> Barostats{{FixPIMDLangevin::MTTK, "MTTK"},
                                     {FixPIMDLangevin::BZP, "BZP"}};
std::map<int, std::string> Ensembles{{FixPIMDLangevin::NVE, "NVE"},
                                     {FixPIMDLangevin::NVT, "NVT"},
                                     {FixPIMDLangevin::NPH, "NPH"},
                                     {FixPIMDLangevin::NPT, "NPT"}};
}    // namespace

namespace {
constexpr int TAG_INTER_REPLICA_COUNT = 10;
constexpr int TAG_INTER_REPLICA_TAGS  = 11;
constexpr int TAG_INTER_REPLICA_VALS  = 12;

constexpr int TAG_RING_PENDING_COUNT = 400;
constexpr int TAG_RING_FOUND_COUNT = 401;
constexpr int TAG_RING_PENDING = 402;
constexpr int TAG_RING_FOUND_TAGS = 403;
constexpr int TAG_RING_FOUND_VALUES = 404;
} // namespace

/* ---------------------------------------------------------------------- */

FixPIMDLangevin::FixPIMDLangevin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), mass(nullptr), plansend(nullptr), planrecv(nullptr), tagsend(nullptr),
    tagrecv(nullptr), bufsend(nullptr), bufrecv(nullptr), bufbeads(nullptr), bufsorted(nullptr),
    bufsortedall(nullptr), counts(nullptr),
    displacements(nullptr), lam(nullptr), M_x2xp(nullptr), M_xp2x(nullptr), M_f2fp(nullptr),
    M_fp2f(nullptr), modeindex(nullptr), tau_k(nullptr), c1_k(nullptr), c2_k(nullptr),
    _omega_k(nullptr), Lan_s(nullptr), Lan_c(nullptr), random(nullptr), xc(nullptr), xcall(nullptr),
    x_unwrap(nullptr), id_pe(nullptr), id_press(nullptr), c_pe(nullptr), c_press(nullptr)
{
  restart_global = 1;
  time_integrate = 1;

  ntotal = 0;
  maxlocal = maxsend = maxunwrap = maxxc = 0;

  sizeplan = 0;

  method = NMPIMD;
  ensemble = NVT;
  integrator = OBABO;
  thermostat = PILE_L;
  barostat = BZP;
  fmass = 1.0;
  np = universe->nworlds;
  inverse_np = 1.0 / np;
  sp = 1.0;
  temp = 298.15;
  Lan_temp = 298.15;
  tau = 1.0;
  tau_p = 1.0;
  Pext = 1.0;
  pdim = 0;
  pilescale = 1.0;
  tstat_flag = 1;
  pstat_flag = 0;
  mapflag = 1;
  removecomflag = 1;
  fmmode = PHYSICAL;
  pstyle = ISO;
  pote = tote = totke = totenthalpy = total_spring_energy = 0.0;
  centroid_vir = vir = vir_ = 0.0;
  ke_bead = se_bead = pe_bead = tote = t_prim = t_vir = t_cv = p_prim = p_md = p_cv = 0.0;

  int seed = -1;

  if (domain->dimension != 3)
    error->universe_all(FLERR, fmt::format("Fix {} requires a 3d system", style));

  for (int i = 0; i < 6; i++) {
    p_flag[i] = 0;
    p_target[i] = 0.0;
  }

  for (int i = 3; i < narg - 1; i += 2) {
    if (strcmp(arg[i], "method") == 0) {
      if (strcmp(arg[i + 1], "nmpimd") == 0)
        method = NMPIMD;
      else if (strcmp(arg[i + 1], "pimd") == 0)
        method = PIMD;
      else
        error->universe_all(FLERR, fmt::format("Unknown method parameter for fix {}", style));
    } else if (strcmp(arg[i], "integrator") == 0) {
      if (strcmp(arg[i + 1], "obabo") == 0)
        integrator = OBABO;
      else if (strcmp(arg[i + 1], "baoab") == 0)
        integrator = BAOAB;
      else
        error->universe_all(FLERR,
                            fmt::format("Unknown integrator parameter for fix {}. Only obabo and "
                                        "baoab integrators are supported!",
                                        style));
    } else if (strcmp(arg[i], "ensemble") == 0) {
      if (strcmp(arg[i + 1], "nve") == 0) {
        ensemble = NVE;
        tstat_flag = 0;
        pstat_flag = 0;
      } else if (strcmp(arg[i + 1], "nvt") == 0) {
        ensemble = NVT;
        tstat_flag = 1;
        pstat_flag = 0;
      } else if (strcmp(arg[i + 1], "nph") == 0) {
        ensemble = NPH;
        tstat_flag = 0;
        pstat_flag = 1;
      } else if (strcmp(arg[i + 1], "npt") == 0) {
        ensemble = NPT;
        tstat_flag = 1;
        pstat_flag = 1;
      } else
        error->universe_all(FLERR,
                            fmt::format("Unknown ensemble parameter for fix {}. Only nve, nvt, "
                                        "nph, and npt ensembles are supported!",
                                        style));
    } else if (strcmp(arg[i], "fmass") == 0) {
      fmass = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (!std::isfinite(fmass) || fmass <= 0.0 || fmass > np)
        error->universe_all(FLERR, fmt::format("Invalid fmass value for fix {}", style));
    } else if (strcmp(arg[i], "sp") == 0) {
      sp = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (sp < 0.0) error->universe_all(FLERR, fmt::format("Invalid sp value for fix {}", style));
    } else if (strcmp(arg[i], "fmmode") == 0) {
      if (strcmp(arg[i + 1], "physical") == 0)
        fmmode = PHYSICAL;
      else if (strcmp(arg[i + 1], "normal") == 0)
        fmmode = NORMAL;
      else
        error->universe_all(FLERR,
                            fmt::format("Unknown fictitious mass mode for fix {}. Only physical "
                                        "mass and normal mode mass are supported!",
                                        style));
    } else if (strcmp(arg[i], "scale") == 0) {
      if (method == PIMD)
        error->universe_all(
            FLERR,
            "The scale parameter of the PILE_L thermostat is not supported for method pimd. Delete "
            "scale parameter if you do want to use method pimd.");
      pilescale = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (pilescale < 0.0)
        error->universe_all(FLERR, fmt::format("Invalid PILE_L scale value for fix {}", style));
    } else if (strcmp(arg[i], "temp") == 0) {
      temp = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (temp < 0.0)
        error->universe_all(FLERR, fmt::format("Invalid temp value for fix {}", style));
    } else if (strcmp(arg[i], "thermostat") == 0) {
      if (strcmp(arg[i + 1], "PILE_L") == 0) {
        thermostat = PILE_L;
        seed = utils::inumeric(FLERR, arg[i + 2], false, lmp);
        i++;
      }
    } else if (strcmp(arg[i], "tau") == 0) {
      tau = utils::numeric(FLERR, arg[i + 1], false, lmp);
    } else if (strcmp(arg[i], "barostat") == 0) {
      if (strcmp(arg[i + 1], "MTTK") == 0) {
        barostat = MTTK;
      } else if (strcmp(arg[i + 1], "BZP") == 0) {
        barostat = BZP;
      } else
        error->universe_all(FLERR, fmt::format("Unknown barostat parameter for fix {}", style));
    } else if (strcmp(arg[i], "iso") == 0) {
      pstyle = ISO;
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      Pext = utils::numeric(FLERR, arg[i + 1], false, lmp);
      p_target[0] = p_target[1] = p_target[2] = Pext;
      pdim = 3;
    } else if (strcmp(arg[i], "aniso") == 0) {
      pstyle = ANISO;
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      Pext = utils::numeric(FLERR, arg[i + 1], false, lmp);
      p_target[0] = p_target[1] = p_target[2] = Pext;
      pdim = 3;
    } else if (strcmp(arg[i], "x") == 0) {
      pstyle = ANISO;
      p_flag[0] = 1;
      p_target[0] = utils::numeric(FLERR, arg[i + 1], false, lmp);
      pdim++;
    } else if (strcmp(arg[i], "y") == 0) {
      pstyle = ANISO;
      p_flag[1] = 1;
      p_target[1] = utils::numeric(FLERR, arg[i + 1], false, lmp);
      pdim++;
    } else if (strcmp(arg[i], "z") == 0) {
      pstyle = ANISO;
      p_flag[2] = 1;
      p_target[2] = utils::numeric(FLERR, arg[i + 1], false, lmp);
      pdim++;
    } else if (strcmp(arg[i], "taup") == 0) {
      tau_p = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (tau_p <= 0.0)
        error->universe_all(FLERR, fmt::format("Invalid tau_p value for fix {}", style));
    } else if (strcmp(arg[i], "fixcom") == 0) {
      if (strcmp(arg[i + 1], "yes") == 0)
        removecomflag = 1;
      else if (strcmp(arg[i + 1], "no") == 0)
        removecomflag = 0;
    } else if (strcmp(arg[i], "") != 0) {
      error->universe_all(FLERR, fmt::format("Unknown keyword {} for fix {}", arg[i], style));
    }
  }

  if (pstat_flag && !pdim)
    error->universe_all(
        FLERR, fmt::format("Must use pressure coupling with {} ensemble", Ensembles[ensemble]));
  if (!pstat_flag && pdim)
    error->universe_all(
        FLERR, fmt::format("Must not use pressure coupling with {} ensemble", Ensembles[ensemble]));

  if (method == PIMD && pstat_flag)
    error->universe_all(FLERR,
                        "Pressure control has not been supported for method pimd yet. Please set "
                        "method to nmpimd.");

  if (method == PIMD && fmmode == NORMAL)
    error->universe_all(
        FLERR, "Normal mode mass is not supported for method pimd. Please set method to nmpimd.");

  /* Initiation */

  global_freq = 1;
  vector_flag = 1;
  if (!pstat_flag) {
    size_vector = 10;
  } else if (pstat_flag) {
    if (pstyle == ISO) {
      size_vector = 15;
    } else if (pstyle == ANISO) {
      size_vector = 17;
    }
  }
  extvector = 1;
  kt = force->boltz * temp;
  if (pstat_flag) FixPIMDLangevin::baro_init();

  // some initilizations

  id_pe = utils::strdup(std::string(id) + "_pimd_pe");
  modify->add_compute(std::string(id_pe) + " all pe");

  id_press = utils::strdup(std::string(id) + "_pimd_press");
  modify->add_compute(std::string(id_press) + " all pressure thermo_temp virial");

  vol0 = domain->xprd * domain->yprd * domain->zprd;

  fixedpoint[0] = 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
  fixedpoint[1] = 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
  fixedpoint[2] = 0.5 * (domain->boxlo[2] + domain->boxhi[2]);
  if (pstat_flag) p_hydro = (p_target[0] + p_target[1] + p_target[2]) / pdim;

  // initialize Marsaglia RNG with processor-unique seed

  if (tstat_flag) {
    if (integrator == BAOAB || integrator == OBABO) {
      Lan_temp = temp;
      random = new RanMars(lmp, seed + universe->me);
    }
  }

  me = comm->me;
  nprocs = comm->nprocs;
  if (nprocs == 1)
    cmode = SINGLE_PROC;
  else
    cmode = MULTI_PROC;

  nprocs_universe = universe->nprocs;
  nreplica = universe->nworlds;
  ireplica = universe->iworld;

  if (nreplica == 1)
    mapflag = 0;
  else
    mapflag = 1;

  int *iroots = new int[nreplica];
  MPI_Group uworldgroup, rootgroup;

  for (int i = 0; i < nreplica; i++) iroots[i] = universe->root_proc[i];
  MPI_Comm_group(universe->uworld, &uworldgroup);
  MPI_Group_incl(uworldgroup, nreplica, iroots, &rootgroup);
  MPI_Comm_create(universe->uworld, rootgroup, &rootworld);
  if (rootgroup != MPI_GROUP_NULL) MPI_Group_free(&rootgroup);
  if (uworldgroup != MPI_GROUP_NULL) MPI_Group_free(&uworldgroup);
  delete[] iroots;

  ntotal = atom->natoms;
  if (atom->nmax > maxlocal) reallocate();
  if (atom->nmax > maxunwrap) reallocate_x_unwrap();
  if (atom->nmax > maxxc) reallocate_xc();
  memory->create(xcall, ntotal * 3, "FixPIMDLangevin:xcall");

}

/* ---------------------------------------------------------------------- */

FixPIMDLangevin::~FixPIMDLangevin()
{
  if (bias_registration) bias_registration->owner = nullptr;
  modify->delete_compute(id_pe);
  modify->delete_compute(id_press);
  delete[] id_pe;
  delete[] id_press;
  delete random;
  delete[] mass;
  delete[] _omega_k;
  delete[] Lan_c;
  delete[] Lan_s;
  delete[] tau_k;
  delete[] c1_k;
  delete[] c2_k;
  delete[] plansend;
  delete[] planrecv;
  delete[] modeindex;
  memory->destroy(xcall);
  if (cmode == SINGLE_PROC) {
    memory->destroy(bufsorted);
    memory->destroy(bufsortedall);
    memory->destroy(counts);
    memory->destroy(displacements);
  }

  memory->sfree(lam);
  memory->destroy(M_x2xp);
  memory->destroy(M_xp2x);
  memory->destroy(xc);
  memory->destroy(x_unwrap);
  memory->destroy(bufsend);
  memory->destroy(bufrecv);
  memory->destroy(tagsend);
  memory->destroy(tagrecv);
  memory->destroy(bufbeads);
  if (rootworld != MPI_COMM_NULL) MPI_Comm_free(&rootworld);
}

/* ---------------------------------------------------------------------- */

int FixPIMDLangevin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= END_OF_STEP;
  mask |= POST_RUN;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::init()
{
  if (bias_registration) validate_partition_topology(true);
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, fmt::format("Fix {} requires an atom map, see atom_modify", style));
  if (atom->tag_consecutive() == 0)
    error->all(FLERR, "Atom IDs must be consecutive for fix {}", style);

  if (universe->me == 0 && universe->uscreen)
    utils::print(universe->uscreen, "Fix {}: initializing Path-Integral ...\n", style);

  // prepare the constants

  masstotal = group->mass(igroup);

  double planck = sp * force->hplanck;
  hbar = planck / MY_2PI;
  beta = 1.0 / (force->boltz * temp);
  double _fbond = 1.0 * np * np / (beta * beta * hbar * hbar);

  omega_np = np / (hbar * beta) * sqrt(force->mvv2e);
  beta_np = 1.0 / force->boltz / temp * inverse_np;
  fbond = _fbond * force->mvv2e;

  if ((universe->me == 0) && (universe->uscreen))
    utils::print(universe->uscreen, "Fix {}: -P/(beta^2 * hbar^2) = {:20.7e} (kcal/mol/A^2)\n\n",
                 style, fbond);

  if (integrator == OBABO) {
    dtf = 0.5 * update->dt * force->ftm2v;
    dtv = 0.5 * update->dt;
    dtv2 = dtv * dtv;
    dtv3 = THIRD * dtv2 * dtv * force->ftm2v;
  } else if (integrator == BAOAB) {
    dtf = 0.5 * update->dt * force->ftm2v;
    dtv = 0.5 * update->dt;
    dtv2 = dtv * dtv;
    dtv3 = THIRD * dtv2 * dtv * force->ftm2v;
  } else {
    error->universe_all(FLERR, fmt::format("Unknown integrator parameter for fix {}", style));
  }

  comm_init();

  // init() runs once per run command, so release the array of the previous one

  delete[] mass;
  mass = new double[atom->ntypes + 1];

  nmpimd_init();

  langevin_init();

  c_pe = modify->get_compute_by_id(id_pe);
  if (!c_pe) {
    error->universe_all(
        FLERR,
        fmt::format("Potential energy compute ID {} for fix {} does not exist", id_pe, style));
  } else {
    if (c_pe->peflag == 0)
      error->universe_all(
          FLERR,
          fmt::format("Compute ID {} for fix {} does not compute potential energy", id_pe, style));
  }

  c_press = modify->get_compute_by_id(id_press);
  if (!c_press) {
    error->universe_all(
        FLERR, fmt::format("Could not find fix {} pressure compute ID {}", style, id_press));
  } else {
    if (c_press->pressflag == 0)
      error->universe_all(
          FLERR,
          fmt::format("Compute ID {} for fix {} does not compute pressure", id_press, style));
  }

  t_prim = t_vir = t_cv = p_prim = p_vir = p_cv = p_md = 0.0;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::setup(int vflag)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  imageint *image = atom->image;
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) domain->unmap(x[i], image[i]);
  }

  if (method == NMPIMD) {
    inter_replica_comm(x);
    if (cmode == SINGLE_PROC)
      nmpimd_transform(bufsortedall, x, M_x2xp[universe->iworld]);
    else if (cmode == MULTI_PROC)
      nmpimd_transform(bufbeads, x, M_x2xp[universe->iworld]);
  } else if (method == PIMD) {
    prepare_coordinates();
    if (!bias_registration) {
      if (cmode == SINGLE_PROC)
        spring_force();
      else if (cmode == MULTI_PROC)
        error->universe_all(FLERR, "Method pimd only supports a single processor per bead");
    }
  } else {
    error->universe_all(
        FLERR,
        fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                    style));
  }
  collect_xc();
  compute_spring_energy();
  compute_t_prim();
  compute_p_prim();
  if (method == NMPIMD) {
    inter_replica_comm(x);
    if (cmode == SINGLE_PROC)
      nmpimd_transform(bufsortedall, x, M_xp2x[universe->iworld]);
    else if (cmode == MULTI_PROC)
      nmpimd_transform(bufbeads, x, M_xp2x[universe->iworld]);
  }
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) domain->unmap_inv(x[i], image[i]);
  }

  post_force(vflag);
  compute_totke();
  if (!bias_pending) end_of_step();
  c_pe->addstep(update->ntimestep + 1);
  c_press->addstep(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::initial_integrate(int /*vflag*/)
{
  require_force_completion();
  int nlocal = atom->nlocal;
  double **x = atom->x;
  imageint *image = atom->image;
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) domain->unmap(x[i], image[i]);
  }
  if (integrator == OBABO) {
    if (tstat_flag) {
      o_step();
      if (removecomflag) remove_com_motion();
      if (pstat_flag) press_o_step();
    }
    if (pstat_flag) {
      compute_totke();
      compute_p_cv();
      press_v_step();
    }
    b_step();
    if (method == NMPIMD) {
      inter_replica_comm(x);
      if (cmode == SINGLE_PROC)
        nmpimd_transform(bufsortedall, x, M_x2xp[universe->iworld]);
      else if (cmode == MULTI_PROC)
        nmpimd_transform(bufbeads, x, M_x2xp[universe->iworld]);
      qc_step();
      a_step();
      qc_step();
      a_step();
    } else if (method == PIMD) {
      q_step();
      q_step();
    } else {
      error->universe_all(
          FLERR,
          fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                      style));
    }
  } else if (integrator == BAOAB) {
    if (pstat_flag) {
      compute_totke();
      compute_p_cv();
      press_v_step();
    }
    b_step();
    if (method == NMPIMD) {
      inter_replica_comm(x);
      if (cmode == SINGLE_PROC)
        nmpimd_transform(bufsortedall, x, M_x2xp[universe->iworld]);
      else if (cmode == MULTI_PROC)
        nmpimd_transform(bufbeads, x, M_x2xp[universe->iworld]);
      qc_step();
      a_step();
    } else if (method == PIMD) {
      q_step();
    } else {
      error->universe_all(
          FLERR,
          fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                      style));
    }
    if (tstat_flag) {
      o_step();
      if (removecomflag) remove_com_motion();
      if (pstat_flag) press_o_step();
    }
    if (method == NMPIMD) {
      qc_step();
      a_step();
    } else if (method == PIMD) {
      q_step();
    } else {
      error->universe_all(
          FLERR,
          fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                      style));
    }
  } else {
    error->universe_all(FLERR,
                        fmt::format("Unknown integrator parameter for fix {}. Only obabo and baoab "
                                    "integrators are supported!",
                                    style));
  }
  collect_xc();

  if (method == NMPIMD) {
    compute_spring_energy();
    compute_t_prim();
    compute_p_prim();
  }

  if (method == NMPIMD) {
    inter_replica_comm(x);
    if (cmode == SINGLE_PROC)
      nmpimd_transform(bufsortedall, x, M_xp2x[universe->iworld]);
    else if (cmode == MULTI_PROC)
      nmpimd_transform(bufbeads, x, M_xp2x[universe->iworld]);
  }

  if (mapflag) {
    for (int i = 0; i < nlocal; i++) { domain->unmap_inv(x[i], image[i]); }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::final_integrate()
{
  require_force_completion();
  if (pstat_flag) {
    compute_totke();
    compute_p_cv();
    press_v_step();
  }
  b_step();
  if (integrator == OBABO) {
    if (tstat_flag) {
      o_step();
      if (removecomflag) remove_com_motion();
      if (pstat_flag) press_o_step();
    }
  } else if (integrator == BAOAB) {

  } else {
    error->universe_all(FLERR, fmt::format("Unknown integrator parameter for fix {}", style));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::prepare_coordinates()
{
  inter_replica_comm(atom->x);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::post_force(int /*flag*/)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  imageint *image = atom->image;
  tagint *tag = atom->tag;

  if (atom->nmax > maxunwrap) reallocate_x_unwrap();
  if (atom->nmax > maxxc) reallocate_xc();
  for (int i = 0; i < nlocal; i++) {
    x_unwrap[i][0] = x[i][0];
    x_unwrap[i][1] = x[i][1];
    x_unwrap[i][2] = x[i][2];
  }
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) { domain->unmap(x_unwrap[i], image[i]); }
  }
  for (int i = 0; i < nlocal; i++) {
    xc[i][0] = xcall[3 * (tag[i] - 1) + 0];
    xc[i][1] = xcall[3 * (tag[i] - 1) + 1];
    xc[i][2] = xcall[3 * (tag[i] - 1) + 2];
  }

  if (bias_registration) {
    if (atom->natoms != ntotal) error->all(FLERR, "External PIMD bias requires a fixed atom count");
    bias_energy = 0.0;
    for (double &value : bias_registration->adapter->virial) value = 0.0;
  }
  compute_vir();
  compute_xf_vir();
  compute_cvir();
  compute_t_vir();
  if (bias_registration) {
    bias_pending = true;
    force_ready = false;
    ++bias_generation;
    return;
  }

  if (method == PIMD) {
    if (mapflag) {
      for (int i = 0; i < nlocal; i++) { domain->unmap(x[i], image[i]); }
    }
    prepare_coordinates();
    spring_force();
    compute_spring_energy();
    compute_t_prim();
    if (mapflag) {
      for (int i = 0; i < nlocal; i++) { domain->unmap_inv(x[i], image[i]); }
    }
  }
  compute_pote();
  if (method == NMPIMD) {
    inter_replica_comm(f);
    if (cmode == SINGLE_PROC)
      nmpimd_transform(bufsortedall, f, M_x2xp[universe->iworld]);
    else if (cmode == MULTI_PROC)
      nmpimd_transform(bufbeads, f, M_x2xp[universe->iworld]);
  }

  c_pe->addstep(update->ntimestep + 1);
  c_press->addstep(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::end_of_step()
{
  require_force_completion();
  compute_totke();
  compute_p_cv();
  compute_tote();
  if (pstat_flag) compute_totenthalpy();
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::collect_xc()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;

  if (method == PIMD) {
    inter_replica_comm(x);
    if (ireplica == 0) {
      for (int i = 0; i < ntotal * 3; i++) xcall[i] = 0.0;
      if (cmode == SINGLE_PROC) {
        for (int i = 0; i < ntotal; i++) {
          for (int d = 0; d < 3; d++) {
            for (int bead = 0; bead < np; bead++)
              xcall[3 * i + d] += bufsortedall[bead * ntotal + i][d] * inverse_np;
          }
        }
      } else {
        for (int i = 0; i < nlocal; i++) {
          const int index = tag[i] - 1;
          for (int d = 0; d < 3; d++) {
            for (int bead = 0; bead < np; bead++)
              xcall[3 * index + d] += bufbeads[bead][3 * i + d] * inverse_np;
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, xcall, ntotal * 3, MPI_DOUBLE, MPI_SUM, world);
      }
    }
  } else {
    if (ireplica == 0) {
      for (int i = 0; i < ntotal; i++) xcall[3 * i + 0] = xcall[3 * i + 1] = xcall[3 * i + 2] = 0.0;

      const double sqrtnp = sqrt((double) np);
      for (int i = 0; i < nlocal; i++) {
        xcall[3 * (tag[i] - 1) + 0] = x[i][0] / sqrtnp;
        xcall[3 * (tag[i] - 1) + 1] = x[i][1] / sqrtnp;
        xcall[3 * (tag[i] - 1) + 2] = x[i][2] / sqrtnp;
      }

      if (cmode == MULTI_PROC)
        MPI_Allreduce(MPI_IN_PLACE, xcall, ntotal * 3, MPI_DOUBLE, MPI_SUM, world);
    }
  }
  MPI_Bcast(xcall, ntotal * 3, MPI_DOUBLE, 0, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::b_step()
{
  // used for both NMPIMD and PIMD
  // For NMPIMD, force only includes the contribution of external potential.
  // For PIMD, force includes the contributions of external potential and spring force.
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *type = atom->type;
  double **v = atom->v;
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double dtfm = dtf / mass[type[i]];
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::qc_step()
{
  // used for NMPIMD
  // evolve the centroid mode
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  double **v = atom->v;
  double oldlo, oldhi;

  if (!pstat_flag) {
    if (universe->iworld == 0) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          x[i][0] += dtv * v[i][0];
          x[i][1] += dtv * v[i][1];
          x[i][2] += dtv * v[i][2];
        }
      }
    }
  } else {
    if (universe->iworld == 0) {
      double expp[3], expq[3];
      if (pstyle == ISO) {
        vw[1] = vw[0];
        vw[2] = vw[0];
      }
      for (int j = 0; j < 3; j++) {
        expq[j] = exp(dtv * vw[j]);
        expp[j] = exp(-dtv * vw[j]);
      }
      if (barostat == BZP) {
        for (int i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            for (int j = 0; j < 3; j++) {
              if (p_flag[j]) {
                x[i][j] = expq[j] * x[i][j] + (expq[j] - expp[j]) / 2. / vw[j] * v[i][j];
                v[i][j] = expp[j] * v[i][j];
              } else {
                x[i][j] += dtv * v[i][j];
              }
            }
          }
        }
        oldlo = domain->boxlo[0];
        oldhi = domain->boxhi[0];

        domain->boxlo[0] = (oldlo - fixedpoint[0]) * expq[0] + fixedpoint[0];
        domain->boxhi[0] = (oldhi - fixedpoint[0]) * expq[0] + fixedpoint[0];

        oldlo = domain->boxlo[1];
        oldhi = domain->boxhi[1];
        domain->boxlo[1] = (oldlo - fixedpoint[1]) * expq[1] + fixedpoint[1];
        domain->boxhi[1] = (oldhi - fixedpoint[1]) * expq[1] + fixedpoint[1];

        oldlo = domain->boxlo[2];
        oldhi = domain->boxhi[2];
        domain->boxlo[2] = (oldlo - fixedpoint[2]) * expq[2] + fixedpoint[2];
        domain->boxhi[2] = (oldhi - fixedpoint[2]) * expq[2] + fixedpoint[2];
      }
    }
    MPI_Barrier(universe->uworld);
    MPI_Bcast(&domain->boxlo[0], 3, MPI_DOUBLE, 0, universe->uworld);
    MPI_Bcast(&domain->boxhi[0], 3, MPI_DOUBLE, 0, universe->uworld);
    domain->set_global_box();
    domain->set_local_box();
    if (force->kspace) force->kspace->setup();
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::a_step()
{
  // used for NMPIMD
  // use analytical solution of harmonic oscillator to evolve the non-centroid modes
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  double **v = atom->v;
  double x0, x1, x2, v0, v1, v2;    // three components of x[i] and v[i]

  if (universe->iworld != 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x0 = x[i][0];
        x1 = x[i][1];
        x2 = x[i][2];
        v0 = v[i][0];
        v1 = v[i][1];
        v2 = v[i][2];
        x[i][0] = Lan_c[universe->iworld] * x0 +
            1.0 / _omega_k[universe->iworld] * Lan_s[universe->iworld] * v0;
        x[i][1] = Lan_c[universe->iworld] * x1 +
            1.0 / _omega_k[universe->iworld] * Lan_s[universe->iworld] * v1;
        x[i][2] = Lan_c[universe->iworld] * x2 +
            1.0 / _omega_k[universe->iworld] * Lan_s[universe->iworld] * v2;
        v[i][0] = -1.0 * _omega_k[universe->iworld] * Lan_s[universe->iworld] * x0 +
            Lan_c[universe->iworld] * v0;
        v[i][1] = -1.0 * _omega_k[universe->iworld] * Lan_s[universe->iworld] * x1 +
            Lan_c[universe->iworld] * v1;
        v[i][2] = -1.0 * _omega_k[universe->iworld] * Lan_s[universe->iworld] * x2 +
            Lan_c[universe->iworld] * v2;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::q_step()
{
  // used for PIMD
  // evolve all beads
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  double **v = atom->v;

  if (!pstat_flag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::baro_init()
{
  vw[0] = vw[1] = vw[2] = vw[3] = vw[4] = vw[5] = 0.0;
  if (pstyle == ISO) {
    W = 3 * (group->count(igroup)) * tau_p * tau_p * np * kt;
  }    // consistent with the definition in i-Pi
  else if (pstyle == ANISO) {
    W = group->count(igroup) * tau_p * tau_p * np * kt;
  }
  Vcoeff = 1.0;
  std::string out = fmt::format("\nInitializing PIMD {:s} barostat...\n", Barostats[barostat]);
  out += fmt::format("  The barostat mass is W = {:.16e}\n", W);
  if (universe->me == 0) utils::logmesg(lmp, out);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::press_v_step()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **f = atom->f;
  double **v = atom->v;
  int *type = atom->type;
  double volume = domain->xprd * domain->yprd * domain->zprd;

  if (pstyle == ISO) {
    if (barostat == BZP) {
      vw[0] += dtv * 3 * (volume * np * (p_cv - p_hydro) / force->nktv2p + Vcoeff / beta_np) / W;
      if (universe->iworld == 0) {
        double dvw_proc = 0.0, dvw = 0.0;
        for (int i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            for (int j = 0; j < 3; j++) {
              dvw_proc += dtv2 * f[i][j] * v[i][j] / W + dtv3 * f[i][j] * f[i][j] / mass[type[i]] / W;
            }
          }
        }
        MPI_Allreduce(&dvw_proc, &dvw, 1, MPI_DOUBLE, MPI_SUM, world);
        vw[0] += dvw;
      }
      MPI_Barrier(universe->uworld);
      MPI_Bcast(&vw[0], 1, MPI_DOUBLE, 0, universe->uworld);
    } else if (barostat == MTTK) {
      double mtk_term1 = 2.0 / group->count(igroup) * totke / 3.0;
      vw[0] += 0.5 * dtv * (volume * np * (p_md - p_hydro) + mtk_term1) / W;
    }
  } else if (pstyle == ANISO) {
    compute_stress_tensor();
    for (int ii = 0; ii < 3; ii++) {
      if (p_flag[ii]) {
        vw[ii] += dtv *
            (volume * np * (stress_tensor[ii] - p_hydro) / force->nktv2p + Vcoeff / beta_np) / W;
        if (universe->iworld == 0) {
          double dvw_proc = 0.0, dvw = 0.0;
          for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
              dvw_proc +=
                  dtv2 * f[i][ii] * v[i][ii] / W + dtv3 * f[i][ii] * f[i][ii] / mass[type[i]] / W;
            }
          }
          MPI_Allreduce(&dvw_proc, &dvw, 1, MPI_DOUBLE, MPI_SUM, world);
          vw[ii] += dvw;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::press_o_step()
{
  if (pstyle == ISO) {
    if (universe->me == 0) vw[0] = c1 * vw[0] + c2 * sqrt(1.0 / W / beta_np) * random->gaussian();
    MPI_Barrier(universe->uworld);
    MPI_Bcast(&vw[0], 1, MPI_DOUBLE, 0, universe->uworld);
  } else if (pstyle == ANISO) {
    if (universe->me == 0) {
      for (int ii = 0; ii < 3; ii++) {
        if (p_flag[ii]) vw[ii] = c1 * vw[ii] + c2 * sqrt(1.0 / W / beta_np) * random->gaussian();
      }
    }
    MPI_Barrier(universe->uworld);
    MPI_Bcast(&vw, 3, MPI_DOUBLE, 0, universe->uworld);
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::langevin_init()
{
  double beta = 1.0 / kt;
  const double _omega_np = np / beta / hbar;
  double _omega_np_dt_half = _omega_np * update->dt * 0.5 / sqrt(fmass);

  delete[] _omega_k;
  delete[] Lan_c;
  delete[] Lan_s;
  _omega_k = new double[np];
  Lan_c = new double[np];
  Lan_s = new double[np];
  if (method == NMPIMD) {
    if (fmmode == PHYSICAL) {
      for (int i = 0; i < np; i++) {
        _omega_k[i] = _omega_np * sqrt(lam[i]) / sqrt(fmass);
        Lan_c[i] = cos(sqrt(lam[i]) * _omega_np_dt_half);
        Lan_s[i] = sin(sqrt(lam[i]) * _omega_np_dt_half);
      }
    } else if (fmmode == NORMAL) {
      for (int i = 0; i < np; i++) {
        _omega_k[i] = _omega_np / sqrt(fmass);
        Lan_c[i] = cos(_omega_np_dt_half);
        Lan_s[i] = sin(_omega_np_dt_half);
      }
    } else {
      error->universe_all(FLERR, "Unknown fmmode setting; only physical and normal are supported!");
    }
  }

  if (tau > 0)
    gamma = 1.0 / tau;
  else
    gamma = np / beta / hbar;

  if (integrator == OBABO)
    c1 = exp(-gamma * 0.5 * update->dt);    // tau is the damping time of the centroid mode.
  else if (integrator == BAOAB)
    c1 = exp(-gamma * update->dt);
  else
    error->universe_all(FLERR,
                        fmt::format("Unknown integrator parameter for fix {}. Only obabo and baoab "
                                    "integrators are supported!",
                                    style));

  c2 = sqrt(1.0 - c1 * c1);    // note that c1 and c2 here only works for the centroid mode.

  if (thermostat == PILE_L) {
    std::string out = "Initializing PI Langevin equation thermostat...\n";
    out += "  Bead ID    |    omega    |    tau    |    c1    |    c2\n";
    if (method == NMPIMD) {
      delete[] tau_k;
      delete[] c1_k;
      delete[] c2_k;
      tau_k = new double[np];
      c1_k = new double[np];
      c2_k = new double[np];
      tau_k[0] = tau;
      c1_k[0] = c1;
      c2_k[0] = c2;
      for (int i = 1; i < np; i++) {
        tau_k[i] = 0.5 / pilescale / _omega_k[i];
        if (integrator == OBABO)
          c1_k[i] = exp(-0.5 * update->dt / tau_k[i]);
        else if (integrator == BAOAB)
          c1_k[i] = exp(-1.0 * update->dt / tau_k[i]);
        else
          error->universe_all(FLERR,
                              fmt::format("Unknown integrator parameter for fix {}. Only obabo and "
                                          "baoab integrators are supported!",
                                          style));
        c2_k[i] = sqrt(1.0 - c1_k[i] * c1_k[i]);
      }
      for (int i = 0; i < np; i++) {
        out += fmt::format("      {:d}     {:.8e} {:.8e} {:.8e} {:.8e}\n", i, _omega_k[i], tau_k[i],
                           c1_k[i], c2_k[i]);
      }
    } else if (method == PIMD) {
      for (int i = 0; i < np; i++) {
        out += fmt::format("      {:d}     {:.8e} {:.8e} {:.8e} {:.8e}\n", i,
                           _omega_np / sqrt(fmass), tau, c1, c2);
      }
    }
    if (thermostat == PILE_L) out += "  PILE_L thermostat successfully initialized!\n";
    out += "\n";
    if (universe->me == 0) utils::logmesg(lmp, out);
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::o_step()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *type = atom->type;
  double beta_np = 1.0 / force->boltz / Lan_temp * inverse_np * force->mvv2e;
  if (thermostat == PILE_L) {
    if (method == NMPIMD) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          atom->v[i][0] = c1_k[universe->iworld] * atom->v[i][0] +
              c2_k[universe->iworld] * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
          atom->v[i][1] = c1_k[universe->iworld] * atom->v[i][1] +
              c2_k[universe->iworld] * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
          atom->v[i][2] = c1_k[universe->iworld] * atom->v[i][2] +
              c2_k[universe->iworld] * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
        }
      }
    } else if (method == PIMD) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          atom->v[i][0] =
              c1 * atom->v[i][0] + c2 * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
          atom->v[i][1] =
              c1 * atom->v[i][1] + c2 * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
          atom->v[i][2] =
              c1 * atom->v[i][2] + c2 * sqrt(1.0 / mass[type[i]] / beta_np) * random->gaussian();
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Normal Mode PIMD
   ------------------------------------------------------------------------- */

void FixPIMDLangevin::nmpimd_init()
{
  // init() calls this once per run, so release what an earlier run allocated

  memory->destroy(M_x2xp);
  memory->destroy(M_xp2x);
  memory->sfree(lam);
  lam = nullptr;

  memory->create(M_x2xp, np, np, "fix_feynman:M_x2xp");
  memory->create(M_xp2x, np, np, "fix_feynman:M_xp2x");

  lam = (double *) memory->smalloc(sizeof(double) * np, "FixPIMDLangevin::lam");

  // Set up  eigenvalues
  for (int i = 0; i < np; i++) {
    double sin_tmp = sin(i * MY_PI / np);
    lam[i] = 4 * sin_tmp * sin_tmp;
  }

  // Set up eigenvectors for degenerated modes
  const double sqrtnp = sqrt((double) np);
  for (int j = 0; j < np; j++) {
    for (int i = 1; i < int(np / 2) + 1; i++) {
      M_x2xp[i][j] = MY_SQRT2 * cos(MY_2PI * double(i) * double(j) / double(np)) / sqrtnp;
    }
    for (int i = int(np / 2) + 1; i < np; i++) {
      M_x2xp[i][j] = MY_SQRT2 * sin(MY_2PI * double(i) * double(j) / double(np)) / sqrtnp;
    }
  }

  // Set up eigenvectors for non-degenerated modes
  for (int i = 0; i < np; i++) {
    M_x2xp[0][i] = 1.0 / sqrtnp;
    if (np % 2 == 0) M_x2xp[np / 2][i] = 1.0 / sqrtnp * powint(-1.0, i);
  }

  // Set up Ut
  for (int i = 0; i < np; i++)
    for (int j = 0; j < np; j++) { M_xp2x[i][j] = M_x2xp[j][i]; }

  // Set up fictitious masses
  int iworld = universe->iworld;
  for (int i = 1; i <= atom->ntypes; i++) {
    mass[i] = atom->mass[i];
    mass[i] *= fmass;
    if (iworld) {
      if (fmmode == PHYSICAL) {
        mass[i] *= 1.0;
      } else if (fmmode == NORMAL) {
        mass[i] *= lam[iworld];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::nmpimd_transform(double **src, double **des, double *vector)
{
  if (cmode == SINGLE_PROC) {
    for (int i = 0; i < ntotal; i++) {
      for (int d = 0; d < 3; d++) {
        bufsorted[i][d] = 0.0;
        for (int j = 0; j < nreplica; j++) {
          bufsorted[i][d] += src[j * ntotal + i][d] * vector[j];
        }
      }
    }
    for (int i = 0; i < ntotal; i++) {
      tagint tagtmp = atom->tag[i];
      for (int d = 0; d < 3; d++) { des[i][d] = bufsorted[tagtmp - 1][d]; }
    }
  } else if (cmode == MULTI_PROC) {
    int n = atom->nlocal;
    int m = 0;

    for (int i = 0; i < n; i++) {
      for (int d = 0; d < 3; d++) {
        des[i][d] = 0.0;
        for (int j = 0; j < np; j++) { des[i][d] += (src[j][m] * vector[j]); }
        m++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::spring_force()
{
  spring_energy = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *_mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  tagint *tagtmp = atom->tag;

  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      const double *xlast;
      const double *xnext;
      if (cmode == SINGLE_PROC) {
        xlast = bufsortedall[x_last * ntotal + tagtmp[i] - 1];
        xnext = bufsortedall[x_next * ntotal + tagtmp[i] - 1];
      } else {
        xlast = &bufbeads[x_last][3 * i];
        xnext = &bufbeads[x_next][3 * i];
      }

      double delx1 = xlast[0] - x[i][0];
      double dely1 = xlast[1] - x[i][1];
      double delz1 = xlast[2] - x[i][2];

      double delx2 = xnext[0] - x[i][0];
      double dely2 = xnext[1] - x[i][1];
      double delz2 = xnext[2] - x[i][2];

      double ff = fbond * _mass[type[i]];

      double dx = delx1 + delx2;
      double dy = dely1 + dely2;
      double dz = delz1 + delz2;

      f[i][0] += dx * ff;
      f[i][1] += dy * ff;
      f[i][2] += dz * ff;

      spring_energy += 0.5 * ff * (delx2 * delx2 + dely2 * dely2 + delz2 * delz2);
    }
  }
}

/* ----------------------------------------------------------------------
   Comm operations
   ------------------------------------------------------------------------- */

void FixPIMDLangevin::comm_init()
{
  if (np != universe->nworlds)
  error->all(FLERR, "Fix pimd/langevin: np must equal universe->nworlds");

  int nlocal = atom->nlocal;
  if (cmode == SINGLE_PROC) {
    // init() calls this once per run, so release what an earlier run allocated
    memory->destroy(counts);
    memory->destroy(displacements);
    memory->create(counts, nreplica, "FixPIMDLangevin:counts");
    memory->create(displacements, nreplica, "FixPIMDLangevin:displacements");
    for (int i = 0; i < nreplica; i++) counts[i] = 3*nlocal;
    displacements[0] = 0;
    for (int i = 0; i < nreplica - 1; i++) displacements[i + 1] = displacements[i] + counts[i];
  }
  if (sizeplan) {
    delete[] plansend;
    delete[] planrecv;
    delete[] modeindex;
  }

  sizeplan = np - 1;
  plansend = new int[sizeplan];
  planrecv = new int[sizeplan];
  modeindex = new int[sizeplan];
  for (int i = 0; i < sizeplan; i++) {

    // send to the (i+1)-th "next" replica, same local rank within that replica
    plansend[i] = universe->me + comm->nprocs * (i + 1);
    if (plansend[i] >= universe->nprocs) plansend[i] -= universe->nprocs;

    // receive from the (i+1)-th "previous" replica, same local rank within that replica
    planrecv[i] = universe->me - comm->nprocs * (i + 1);
    if (planrecv[i] < 0) planrecv[i] += universe->nprocs;

    // where to store what we receive this round:
    // this is the replica index you are pulling from in this step
    modeindex[i] = (universe->iworld + i + 1) % universe->nworlds;
  }

  x_next = (universe->iworld + 1 + universe->nworlds) % (universe->nworlds);
  x_last = (universe->iworld - 1 + universe->nworlds) % (universe->nworlds);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::reallocate_xc()
{
  maxxc = std::max(1, atom->nmax);
  memory->destroy(xc);
  memory->create(xc, maxxc, 3, "FixPIMDLangevin:xc");
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::reallocate_x_unwrap()
{
  maxunwrap = std::max(1, atom->nmax);
  memory->destroy(x_unwrap);
  memory->create(x_unwrap, maxunwrap, 3, "FixPIMDLangevin:x_unwrap");
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::reallocate()
{
  maxlocal = std::max(1, atom->nmax);
  maxsend = std::max(maxsend, maxlocal);
  ntotal = atom->natoms;
  if (cmode == SINGLE_PROC) {
    memory->destroy(bufsorted);
    memory->destroy(bufsortedall);
    memory->create(bufsorted, std::max(1, ntotal), 3, "FixPIMDLangevin:bufsorted");
    memory->create(bufsortedall, std::max(1, nreplica * ntotal), 3, "FixPIMDLangevin:bufsortedall");
  } else if (cmode == MULTI_PROC) {
    memory->destroy(bufsend);
    memory->destroy(bufrecv);
    memory->destroy(tagsend);
    memory->destroy(tagrecv);
    memory->destroy(bufbeads);
    memory->create(bufsend, maxsend * 3, "FixPIMDLangevin:bufsend");
    memory->create(bufrecv, maxlocal * 3, "FixPIMDLangevin:bufrecv");
    memory->create(tagsend, maxsend, "FixPIMDLangevin:tagsend");
    memory->create(tagrecv, maxlocal, "FixPIMDLangevin:tagrecv");
    memory->create(bufbeads, nreplica, maxlocal * 3, "FixPIMDLangevin:bufrecv");
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::inter_replica_comm(double **ptr)
{
  if (!bufsend && !bufsorted)
    reallocate();
  else if (atom->nmax > maxlocal)
    reallocate();
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int i, m;

  // communicate values from the other beads
  if (cmode == SINGLE_PROC) {
    m = 0;
    for (i = 0; i < nlocal; i++) {
      tagint tagtmp = tag[i];
      bufsorted[tagtmp - 1][0] = ptr[i][0];
      bufsorted[tagtmp - 1][1] = ptr[i][1];
      bufsorted[tagtmp - 1][2] = ptr[i][2];
      m++;
    }
    MPI_Allgatherv(bufsorted[0], 3 * m, MPI_DOUBLE, bufsortedall[0], counts, displacements,
                   MPI_DOUBLE, universe->uworld);
  } else if (cmode == MULTI_PROC) {
    // buffers are (re)allocated as needed in reallocate()
    // copy local values
    for (i = 0; i < nlocal; i++) {
      bufbeads[ireplica][3 * i + 0] = ptr[i][0];
      bufbeads[ireplica][3 * i + 1] = ptr[i][1];
      bufbeads[ireplica][3 * i + 2] = ptr[i][2];
    }

    // Loop over replica comm plans
    for (int iplan = 0; iplan < sizeplan; iplan++) {

      // 1) exchange local counts between the paired ranks in universe->uworld
      int nsend = 0;
      MPI_Sendrecv((void *) &nlocal, 1, MPI_INT, plansend[iplan], TAG_INTER_REPLICA_COUNT,
                   (void *) &nsend, 1, MPI_INT, planrecv[iplan], TAG_INTER_REPLICA_COUNT,
                   universe->uworld, MPI_STATUS_IGNORE);

      // 2) ensure buffers sized for nsend
      if (nsend > maxsend) {
        maxsend = nsend + 200;
        tagsend = (tagint *) memory->srealloc(tagsend, sizeof(tagint) * maxsend,
                                              "FixPIMDLangevin:tagsend");
        bufsend = (double *) memory->srealloc(bufsend, sizeof(double) * 3 * maxsend,
                                              "FixPIMDLangevin:bufsend");
      }

      // 3) exchange tags:
      //    send my local tags (atom->tag[0..nlocal-1])
      //    receive remote rank's local tags into tagsend[0..nsend-1]
      MPI_Sendrecv((void *) atom->tag, nlocal, MPI_LMP_TAGINT, plansend[iplan],
                   TAG_INTER_REPLICA_TAGS, (void *) tagsend, nsend, MPI_LMP_TAGINT, planrecv[iplan],
                   TAG_INTER_REPLICA_TAGS, universe->uworld, MPI_STATUS_IGNORE);

      // 4) pack ptr for the tags the remote rank needs from me
      //    For each received tag, find my local index and copy ptr[index][0..2]
      std::vector<int> miss_idx;
      std::vector<tagint> miss_tag;
      miss_idx.reserve(nsend);
      miss_tag.reserve(nsend);

      for (int i = 0; i < nsend; i++) {
        const int idx = atom->map(tagsend[i]);
        if (idx >= 0 && idx < nlocal) {
          bufsend[3 * i + 0] = ptr[idx][0];
          bufsend[3 * i + 1] = ptr[idx][1];
          bufsend[3 * i + 2] = ptr[idx][2];
        } else {
          miss_idx.push_back(i);             // remember which slot in bufsend needs collect
          miss_tag.push_back(tagsend[i]);    // remember which tag that slot corresponds to
        }
      }

      // 5) collect missing tags within this world (local-only claiming)
      // Every rank must forward ring requests, even if its own list is empty.
      int has_missing_tags = miss_tag.empty() ? 0 : 1;
      MPI_Allreduce(MPI_IN_PLACE, &has_missing_tags, 1, MPI_INT, MPI_MAX, world);

      std::vector<double> collected_values;
      if (has_missing_tags) ring_collect(miss_tag, ptr, collected_values);

      if (!miss_tag.empty()) {
        for (int k = 0; k < static_cast<int>(miss_tag.size()); k++) {
          const int i = miss_idx[k];
          bufsend[3 * i + 0] = collected_values[3 * k + 0];
          bufsend[3 * i + 1] = collected_values[3 * k + 1];
          bufsend[3 * i + 2] = collected_values[3 * k + 2];
        }
      }

      // 6) exchange packed x/f buffers:
      //    - send bufsend (3*nsend) to planrecv[iplan]
      //    - receive bufrecv (3*nlocal) from plansend[iplan]
      //
      // Exchange values with the corresponding rank in the target bead.
      MPI_Sendrecv((void *) bufsend, 3 * nsend, MPI_DOUBLE, planrecv[iplan], TAG_INTER_REPLICA_VALS,
                   (void *) bufrecv, 3 * nlocal, MPI_DOUBLE, plansend[iplan],
                   TAG_INTER_REPLICA_VALS, universe->uworld, MPI_STATUS_IGNORE);

      // 7) store received x/f for this plan into bufbeads[modeindex[iplan]]
      memcpy(bufbeads[modeindex[iplan]], bufrecv, sizeof(double) * 3 * nlocal);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::ring_collect(const std::vector<tagint> &requested_tags, double **source,
                                   std::vector<double> &requested_values)
{
  const int me = comm->me;
  const int nprocs = comm->nprocs;
  const int next = (me + 1) % nprocs;
  const int prev = (me - 1 + nprocs) % nprocs;
  const int nlocal = atom->nlocal;
  tagint unused_tag = 0;
  double unused_value = 0.0;
  const auto tag_buffer = [&unused_tag](std::vector<tagint> &values) {
    return values.empty() ? &unused_tag : values.data();
  };
  const auto value_buffer = [&unused_value](std::vector<double> &values) {
    return values.empty() ? &unused_value : values.data();
  };

  int too_many = requested_tags.size() > static_cast<std::size_t>(MAXSMALLINT / 3);
  MPI_Allreduce(MPI_IN_PLACE, &too_many, 1, MPI_INT, MPI_MAX, lmp->world);
  if (too_many) lmp->error->all(FLERR, "Fix {} has too many atom vector requests", style);

  std::vector<tagint> pending(requested_tags);
  std::vector<tagint> found_tags;
  std::vector<double> found_values;
  found_tags.reserve(pending.size());
  found_values.reserve(3 * pending.size());

  for (int hop = 0; hop < nprocs; hop++) {
    const int send_pending = static_cast<int>(pending.size());
    const int send_found = static_cast<int>(found_tags.size());
    int recv_pending, recv_found;

    MPI_Sendrecv(&send_pending, 1, MPI_INT, next, TAG_RING_PENDING_COUNT, &recv_pending, 1, MPI_INT,
                 prev, TAG_RING_PENDING_COUNT, lmp->world, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_found, 1, MPI_INT, next, TAG_RING_FOUND_COUNT, &recv_found, 1, MPI_INT, prev,
                 TAG_RING_FOUND_COUNT, lmp->world, MPI_STATUS_IGNORE);

    std::vector<tagint> incoming_pending(recv_pending);
    std::vector<tagint> incoming_found_tags(recv_found);
    std::vector<double> incoming_found_values(3 * static_cast<std::size_t>(recv_found));

    MPI_Sendrecv(tag_buffer(pending), send_pending, MPI_LMP_TAGINT, next, TAG_RING_PENDING,
                 tag_buffer(incoming_pending), recv_pending, MPI_LMP_TAGINT, prev, TAG_RING_PENDING,
                 lmp->world, MPI_STATUS_IGNORE);
    MPI_Sendrecv(tag_buffer(found_tags), send_found, MPI_LMP_TAGINT, next, TAG_RING_FOUND_TAGS,
                 tag_buffer(incoming_found_tags), recv_found, MPI_LMP_TAGINT, prev,
                 TAG_RING_FOUND_TAGS, lmp->world, MPI_STATUS_IGNORE);
    MPI_Sendrecv(value_buffer(found_values), 3 * send_found, MPI_DOUBLE, next,
                 TAG_RING_FOUND_VALUES, value_buffer(incoming_found_values), 3 * recv_found,
                 MPI_DOUBLE, prev, TAG_RING_FOUND_VALUES, lmp->world, MPI_STATUS_IGNORE);

    std::vector<tagint> still_pending;
    still_pending.reserve(incoming_pending.size());
    for (tagint tag : incoming_pending) {
      const int index = atom->map(tag);
      if (index >= 0 && index < nlocal) {
        incoming_found_tags.push_back(tag);
        incoming_found_values.push_back(source[index][0]);
        incoming_found_values.push_back(source[index][1]);
        incoming_found_values.push_back(source[index][2]);
      } else {
        still_pending.push_back(tag);
      }
    }

    pending.swap(still_pending);
    found_tags.swap(incoming_found_tags);
    found_values.swap(incoming_found_values);
  }

  if (!pending.empty())
    lmp->error->universe_one(FLERR,
                             fmt::format("Fix {} could not find atom ID {} in partition {}", style,
                                         pending[0], universe->iworld));

  requested_values.resize(3 * requested_tags.size());
  for (std::size_t i = 0; i < requested_tags.size(); i++) {
    const auto iter = std::find(found_tags.begin(), found_tags.end(), requested_tags[i]);
    if (iter == found_tags.end())
      lmp->error->universe_one(FLERR,
                               fmt::format("Fix {} could not collect atom ID {} in partition {}",
                                           style, requested_tags[i], universe->iworld));
    const std::size_t index = std::distance(found_tags.begin(), iter);
    requested_values[3 * i] = found_values[3 * index];
    requested_values[3 * i + 1] = found_values[3 * index + 1];
    requested_values[3 * i + 2] = found_values[3 * index + 2];
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::remove_com_motion()
{
  if (method == NMPIMD) {
    if (universe->iworld == 0) {
      double **v = atom->v;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;
      if (dynamic) masstotal = group->mass(igroup);
      double vcm[3];
      group->vcm(igroup, masstotal, vcm);
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          v[i][0] -= vcm[0];
          v[i][1] -= vcm[1];
          v[i][2] -= vcm[2];
        }
      }
    }
  } else if (method == PIMD) {
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (dynamic) masstotal = group->mass(igroup);
    double vcm[3];
    group->vcm(igroup, masstotal, vcm);
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] -= vcm[0];
        v[i][1] -= vcm[1];
        v[i][2] -= vcm[2];
      }
    }
  } else {
    error->all(
        FLERR,
        fmt::format("Unknown method for fix {}. Only nmpimd and pimd are supported!", style));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_xf_vir()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double xf = 0.0;
  vir_ = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (int j = 0; j < 3; j++) { xf += x_unwrap[i][j] * atom->f[i][j]; }
    }
  }
  MPI_Allreduce(&xf, &vir_, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_cvir()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double xcf = 0.0;
  centroid_vir = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (int j = 0; j < 3; j++) { xcf += (x_unwrap[i][j] - xc[i][j]) * atom->f[i][j]; }
    }
  }
  MPI_Allreduce(&xcf, &centroid_vir, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  if (pstyle == ANISO) {
    for (int i = 0; i < 6; i++) c_vir_tensor[i] = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        c_vir_tensor[0] += (x_unwrap[i][0] - xc[i][0]) * atom->f[i][0];
        c_vir_tensor[1] += (x_unwrap[i][1] - xc[i][1]) * atom->f[i][1];
        c_vir_tensor[2] += (x_unwrap[i][2] - xc[i][2]) * atom->f[i][2];
        c_vir_tensor[3] += (x_unwrap[i][0] - xc[i][0]) * atom->f[i][1];
        c_vir_tensor[4] += (x_unwrap[i][0] - xc[i][0]) * atom->f[i][2];
        c_vir_tensor[5] += (x_unwrap[i][1] - xc[i][1]) * atom->f[i][2];
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &c_vir_tensor, 6, MPI_DOUBLE, MPI_SUM, universe->uworld);
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_vir()
{
  double volume = domain->xprd * domain->yprd * domain->zprd;
  c_press->compute_vector();
  virial[0] = c_press->vector[0] * volume;
  virial[1] = c_press->vector[1] * volume;
  virial[2] = c_press->vector[2] * volume;
  virial[3] = c_press->vector[3] * volume;
  virial[4] = c_press->vector[4] * volume;
  virial[5] = c_press->vector[5] * volume;
  for (int i = 0; i < 6; i++) virial[i] /= universe->procs_per_world[universe->iworld];
  double vir_bead = (virial[0] + virial[1] + virial[2]);
  MPI_Allreduce(&vir_bead, &vir, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  MPI_Allreduce(MPI_IN_PLACE, &virial[0], 6, MPI_DOUBLE, MPI_SUM, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_stress_tensor()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *type = atom->type;
  if (universe->iworld == 0) {
    double inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    for (int i = 0; i < 6; i++) ke_tensor[i] = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        ke_tensor[0] += 0.5 * mass[type[i]] * atom->v[i][0] * atom->v[i][0] * force->mvv2e;
        ke_tensor[1] += 0.5 * mass[type[i]] * atom->v[i][1] * atom->v[i][1] * force->mvv2e;
        ke_tensor[2] += 0.5 * mass[type[i]] * atom->v[i][2] * atom->v[i][2] * force->mvv2e;
        ke_tensor[3] += 0.5 * mass[type[i]] * atom->v[i][0] * atom->v[i][1] * force->mvv2e;
        ke_tensor[4] += 0.5 * mass[type[i]] * atom->v[i][0] * atom->v[i][2] * force->mvv2e;
        ke_tensor[5] += 0.5 * mass[type[i]] * atom->v[i][1] * atom->v[i][2] * force->mvv2e;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &ke_tensor, 6, MPI_DOUBLE, MPI_SUM, world);
    for (int i = 0; i < 6; i++) {
      stress_tensor[i] =
          inv_volume * ((2 * ke_tensor[i] - c_vir_tensor[i]) * force->nktv2p + virial[i]) / np;
    }
  }
  MPI_Bcast(&stress_tensor, 6, MPI_DOUBLE, 0, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_totke()
{
  double kine = 0.0;
  totke = ke_bead = 0.0;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *type = atom->type;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (int j = 0; j < 3; j++) { kine += 0.5 * mass[type[i]] * atom->v[i][j] * atom->v[i][j]; }
    }
  }
  kine *= force->mvv2e;
  MPI_Allreduce(&kine, &ke_bead, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ke_bead, &totke, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  totke /= universe->procs_per_world[universe->iworld];
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_spring_energy()
{
  if (method == NMPIMD) {
    spring_energy = 0.0;
    total_spring_energy = se_bead = 0.0;

    double **x = atom->x;
    double *_mass = atom->mass;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int *mask = atom->mask;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        spring_energy += 0.5 * _mass[type[i]] * fbond * lam[universe->iworld] *
            (x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2] * x[i][2]);
      }
    }
    MPI_Allreduce(&spring_energy, &se_bead, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&se_bead, &total_spring_energy, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    total_spring_energy /= universe->procs_per_world[universe->iworld];
  } else if (method == PIMD) {
    total_spring_energy = se_bead = 0.0;
    MPI_Allreduce(&spring_energy, &se_bead, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&se_bead, &total_spring_energy, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    total_spring_energy /= universe->procs_per_world[universe->iworld];
  } else {
    error->universe_all(
        FLERR,
        fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                    style));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_pote()
{
  pe_bead = 0.0;
  pote = 0.0;
  c_pe->compute_scalar();
  pe_bead = c_pe->scalar;
  // The adapter reports physical B; this beta/P Hamiltonian contains P*B.
  if (bias_registration) {
    const auto *adapter = bias_registration->adapter;
    const bool tallied = adapter->energy_global_flag && adapter->thermo_energy;
    pe_bead += (np - (tallied ? 1 : 0)) * bias_energy;
  }
  double pot_energy_partition = pe_bead / universe->procs_per_world[universe->iworld];
  MPI_Allreduce(&pot_energy_partition, &pote, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_tote()
{
  tote = totke + pote + total_spring_energy;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_t_prim()
{
  t_prim = 1.5 * group->count(igroup) * np * force->boltz * temp - total_spring_energy * inverse_np;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_t_vir()
{
  t_vir = -0.5 * inverse_np * vir_;
  t_cv = 1.5 * group->count(igroup) * force->boltz * temp - 0.5 * inverse_np * centroid_vir;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_p_prim()
{
  double inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
  p_prim = group->count(igroup) * np * force->boltz * temp * inv_volume -
      1.0 / 1.5 * inv_volume * total_spring_energy;
  p_prim *= force->nktv2p;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_p_cv()
{
  double inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
  p_md = THIRD * inv_volume * (totke + vir);
  if (method == NMPIMD) {
    if (universe->iworld == 0) {
      p_cv = THIRD * inv_volume * ((2.0 * ke_bead - centroid_vir) * force->nktv2p + vir) / np;
    }
    MPI_Bcast(&p_cv, 1, MPI_DOUBLE, 0, universe->uworld);
  } else if (method == PIMD) {
    p_cv = THIRD * inv_volume * ((2.0 * totke / np - centroid_vir) * force->nktv2p + vir) / np;
  } else {
    error->universe_all(
        FLERR,
        fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                    style));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::compute_totenthalpy()
{
  double volume = domain->xprd * domain->yprd * domain->zprd;
  if (barostat == BZP) {
    if (pstyle == ISO) {
      totenthalpy = tote + 0.5 * W * vw[0] * vw[0] * inverse_np + p_hydro * volume / force->nktv2p -
          Vcoeff * kt * log(volume);
    } else if (pstyle == ANISO) {
      totenthalpy = tote + 0.5 * W * vw[0] * vw[0] * inverse_np +
          0.5 * W * vw[1] * vw[1] * inverse_np + 0.5 * W * vw[2] * vw[2] * inverse_np +
          p_hydro * volume / force->nktv2p - Vcoeff * kt * log(volume);
    }
  } else if (barostat == MTTK)
    totenthalpy = tote + 1.5 * W * vw[0] * vw[0] * inverse_np + p_hydro * (volume - vol0);
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixPIMDLangevin::write_restart(FILE *fp)
{
  int nsize = size_restart_global();

  double *list;
  memory->create(list, nsize, "FixPIMDLangevin:list");

  pack_restart_data(list);

  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), nsize, fp);
  }

  memory->destroy(list);
}
/* ---------------------------------------------------------------------- */

int FixPIMDLangevin::size_restart_global()
{
  int nsize = 6;

  return nsize;
}

/* ---------------------------------------------------------------------- */

int FixPIMDLangevin::pack_restart_data(double *list)
{
  int n = 0;
  for (int i = 0; i < 6; i++) list[n++] = vw[i];
  return n;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;
  for (int i = 0; i < 6; i++) vw[i] = list[n++];
}

/* ---------------------------------------------------------------------- */

double FixPIMDLangevin::compute_vector(int n)
{
  require_force_completion();
  if (n == 0) return ke_bead;
  if (n == 1) return se_bead;
  if (n == 2) return pe_bead;
  if (n == 3) return tote;
  if (n == 4) return t_prim;
  if (n == 5) return t_vir;
  if (n == 6) return t_cv;
  if (n == 7) return p_prim;
  if (n == 8) return p_md;
  if (n == 9) return p_cv;

  if (pstat_flag) {
    double volume = domain->xprd * domain->yprd * domain->zprd;
    if (pstyle == ISO) {
      if (n == 10) return vw[0];
      if (barostat == BZP) {
        if (n == 11) return 0.5 * W * vw[0] * vw[0];
      } else if (barostat == MTTK) {
        if (n == 11) return 1.5 * W * vw[0] * vw[0];
      }
      if (n == 12) { return np * Pext * volume / force->nktv2p; }
      if (n == 13) { return -Vcoeff * np * kt * log(volume); }
      if (n == 14) return totenthalpy;
    } else if (pstyle == ANISO) {
      if (n == 10) return vw[0];
      if (n == 11) return vw[1];
      if (n == 12) return vw[2];
      if (n == 13) return 0.5 * W * (vw[0] * vw[0] + vw[1] * vw[1] + vw[2] * vw[2]);
      if (n == 14) { return np * Pext * volume / force->nktv2p; }
      if (n == 15) {
        double volume = domain->xprd * domain->yprd * domain->zprd;
        return -Vcoeff * np * kt * log(volume);
      }
      if (n == 16) return totenthalpy;
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::validate_partition_topology(bool require_consecutive_tags)
{
  int min_ranks, max_ranks;
  MPI_Allreduce(&comm->nprocs, &min_ranks, 1, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(&comm->nprocs, &max_ranks, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (min_ranks != max_ranks)
    error->all(FLERR, "External PIMD bias requires equal MPI ranks per partition");
  if (!atom->mass || atom->rmass_flag)
    error->all(FLERR, "External PIMD bias requires masses defined by atom type");
  bigint min_natoms, max_natoms;
  MPI_Allreduce(&atom->natoms, &min_natoms, 1, MPI_LMP_BIGINT, MPI_MIN, universe->uworld);
  MPI_Allreduce(&atom->natoms, &max_natoms, 1, MPI_LMP_BIGINT, MPI_MAX, universe->uworld);
  if (min_natoms != max_natoms)
    error->all(FLERR,
               fmt::format("Fix {} requires the same number of atoms in every partition", style));

  double cell[9] = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2],
                    domain->boxhi[0], domain->boxhi[1], domain->boxhi[2],
                    domain->xy,       domain->xz,       domain->yz};
  double min_cell[9], max_cell[9];
  MPI_Allreduce(cell, min_cell, 9, MPI_DOUBLE, MPI_MIN, universe->uworld);
  MPI_Allreduce(cell, max_cell, 9, MPI_DOUBLE, MPI_MAX, universe->uworld);
  int cell_mismatch = 0;
  for (int i = 0; i < 9; i++)
    if (min_cell[i] != max_cell[i]) cell_mismatch = 1;

  int cell_flags[8] = {domain->triclinic,      domain->triclinic_general, domain->boundary[0][0],
                       domain->boundary[0][1], domain->boundary[1][0],    domain->boundary[1][1],
                       domain->boundary[2][0], domain->boundary[2][1]};
  int min_flags[8], max_flags[8];
  MPI_Allreduce(cell_flags, min_flags, 8, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(cell_flags, max_flags, 8, MPI_INT, MPI_MAX, universe->uworld);
  for (int i = 0; i < 8; i++)
    if (min_flags[i] != max_flags[i]) cell_mismatch = 1;
  if (cell_mismatch)
    error->all(FLERR,
               fmt::format("Fix {} requires the same simulation cell in every partition", style));

  int min_ntypes, max_ntypes;
  MPI_Allreduce(&atom->ntypes, &min_ntypes, 1, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(&atom->ntypes, &max_ntypes, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (min_ntypes != max_ntypes)
    error->all(FLERR,
               fmt::format("Fix {} requires the same atom masses in every partition", style));

  std::vector<double> min_mass(atom->ntypes), max_mass(atom->ntypes);
  MPI_Allreduce(atom->mass + 1, min_mass.data(), atom->ntypes, MPI_DOUBLE, MPI_MIN,
                universe->uworld);
  MPI_Allreduce(atom->mass + 1, max_mass.data(), atom->ntypes, MPI_DOUBLE, MPI_MAX,
                universe->uworld);
  for (int i = 0; i < atom->ntypes; i++)
    if (min_mass[i] != max_mass[i])
      error->all(FLERR,
                 fmt::format("Fix {} requires the same atom masses in every partition", style));

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, fmt::format("Fix {} requires an atom map, see atom_modify", style));
  if (require_consecutive_tags && atom->tag_consecutive() == 0)
    error->all(FLERR, "Atom IDs must be consecutive for fix {}", style);
  check_atom_consistency();
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::check_atom_consistency()
{
  if (atom->natoms >= MAXSMALLINT)
    lmp->error->all(FLERR, "Fix {} atom type consistency check supports fewer than {} atoms", style,
                    MAXSMALLINT);
  const int natoms = static_cast<int>(atom->natoms);

  std::vector<tagint> tags(natoms);
  if (universe->iworld == 0) {
    std::vector<int> counts(comm->nprocs), displacements(comm->nprocs);
    MPI_Allgather(&atom->nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, lmp->world);
    for (int i = 1; i < comm->nprocs; i++) displacements[i] = displacements[i - 1] + counts[i - 1];
    MPI_Allgatherv(atom->tag, atom->nlocal, MPI_LMP_TAGINT, tags.data(), counts.data(),
                   displacements.data(), MPI_LMP_TAGINT, lmp->world);
    std::sort(tags.begin(), tags.end());
  }
  MPI_Bcast(tags.data(), natoms, MPI_LMP_TAGINT, 0, universe->uworld);

  std::vector<int> types(natoms, 0), members(natoms, 0);
  for (int i = 0; i < natoms; i++) {
    const int index = atom->map(tags[i]);
    if (index >= 0 && index < atom->nlocal) {
      types[i] = atom->type[index];
      members[i] = (atom->mask[index] & groupbit) ? 1 : 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, types.data(), natoms, MPI_INT, MPI_SUM, lmp->world);
  MPI_Allreduce(MPI_IN_PLACE, members.data(), natoms, MPI_INT, MPI_SUM, lmp->world);

  std::vector<int> reference_types(types);
  MPI_Bcast(reference_types.data(), natoms, MPI_INT, 0, universe->uworld);
  int mismatch = (types != reference_types) ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (mismatch)
    lmp->error->all(
        FLERR, "Fix {} requires the same atom types for every atom ID in every partition", style);

  std::vector<int> reference_members(members);
  MPI_Bcast(reference_members.data(), natoms, MPI_INT, 0, universe->uworld);
  mismatch = (members != reference_members) ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (mismatch)
    lmp->error->all(
        FLERR, "Fix {} requires the same group membership for every atom ID in every partition",
        style);
}

/* ---------------------------------------------------------------------- */

std::shared_ptr<PIMDBiasRegistration>
FixPIMDLangevin::register_bias(Fix *adapter, PIMDBiasMode mode, bool contraction)
{
  const auto capabilities = bias_capabilities();
  const bool supported = mode == PIMDBiasMode::COORDINATE_MEAN ? capabilities.coordinate_mean
                                                               : capabilities.bead_modes;
  int settings[2] = {static_cast<int>(mode), contraction ? 1 : 0};
  int minimum[2], maximum[2];
  MPI_Allreduce(settings, minimum, 2, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(settings, maximum, 2, MPI_INT, MPI_MAX, universe->uworld);
  if ((minimum[0] != maximum[0]) || (minimum[1] != maximum[1]))
    error->all(FLERR, "External PIMD bias settings must match in all partitions");
  int invalid = !supported ? 1 : 0;
  if (contraction && !capabilities.contraction) invalid = 2;
  if (bias_registration &&
      ((bias_registration->adapter != adapter) || (bias_registration->mode != mode) ||
       (bias_registration->contraction != contraction)))
    invalid = 3;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid == 3)
    error->all(FLERR, "Fix {} already has a different external bias registration", style);
  if (invalid == 2)
    error->all(FLERR, "Fix {} external bias contraction requires a supported PIMD ensemble", style);
  if (invalid == 1)
    error->all(FLERR, "Fix {} does not support the requested external PIMD bias mode", style);

  // Publish a token only after validation, so a failed init leaves no registration.
  validate_partition_topology(true);
  if (pstat_flag && domain->triclinic)
    error->all(FLERR, "External PIMD bias with pressure control requires an orthogonal cell");
  if (pstat_flag && barostat != BZP)
    error->all(FLERR, "External PIMD bias with pressure control requires barostat BZP");
  for (const auto &fix : modify->get_fix_list())
    if ((fix != this) && fix->box_change)
      error->all(FLERR, "External PIMD bias cannot be combined with another cell-changing fix");
  validate_bias_order(adapter);
  if (!bias_registration) {
    bias_registration = std::make_shared<PIMDBiasRegistration>();
    bias_registration->owner = this;
    bias_registration->adapter = adapter;
    bias_registration->mode = mode;
    bias_registration->contraction = contraction;
  }
  return bias_registration;
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::unregister_bias(PIMDBiasRegistration &registration)
{
  if (bias_registration.get() != &registration) return;
  registration.owner = nullptr;
  registration.adapter = nullptr;
  bias_registration.reset();
  bias_pending = false;
  bias_energy = 0.0;
}

/* ---------------------------------------------------------------------- */

PIMDBiasCoordinates
FixPIMDLangevin::bias_coordinates(const PIMDBiasRegistration &registration) const
{
  int state_error = !bias_pending ? 1 : 0;
  if ((bias_registration.get() != &registration) || (registration.owner != this)) state_error = 2;
  MPI_Allreduce(MPI_IN_PLACE, &state_error, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (state_error == 2) error->all(FLERR, "Stale external PIMD bias registration");
  if (state_error == 1) error->all(FLERR, "No pending external PIMD bias force evaluation");
  return {bias_generation, update->ntimestep, atom->nlocal, atom->tag, atom->x, x_unwrap, xcall};
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::submit_bias(const PIMDBiasRegistration &registration,
                                  const PIMDBiasSubmission &submission, double *dynamical_virial)
{
  int state_error = 0;
  if (!bias_pending) state_error = 1;
  if (submission.generation != bias_generation) state_error = 2;
  if ((bias_registration.get() != &registration) || (registration.owner != this)) state_error = 3;
  MPI_Allreduce(MPI_IN_PLACE, &state_error, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (state_error == 3) error->all(FLERR, "Stale external PIMD bias registration");
  if (state_error == 2) error->all(FLERR, "Stale external PIMD bias force evaluation");
  if (state_error == 1) error->all(FLERR, "Duplicate external PIMD bias force submission");
  int invalid = (submission.nlocal != atom->nlocal) || !submission.virial || !dynamical_virial ||
      (atom->nlocal && !submission.force) || !std::isfinite(submission.energy);
  if (!invalid) {
    for (int i = 0; i < 3 * atom->nlocal; ++i)
      if (!std::isfinite(submission.force[i])) invalid = 1;
    for (int i = 0; i < 6; ++i)
      if (!std::isfinite(submission.virial[i])) invalid = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Invalid external PIMD bias submission");
  bias_energy = submission.energy;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) atom->f[i][d] += np * submission.force[3 * i + d];
  for (int d = 0; d < 6; ++d) dynamical_virial[d] = np * submission.virial[d];

  // Keep the physical kinetic estimators captured before external forces.
  // Refresh only biased pressure from Cartesian forces, before adding springs
  // or changing the force representation.
  c_press->invoked_vector = -1;
  c_pe->invoked_scalar = -1;
  compute_vir();
  compute_cvir();
  bias_pending = false;
  complete_force_evaluation();
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::require_force_completion() const
{
  if (!force_ready) error->all(FLERR, "Missing external PIMD bias force submission");
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::post_run()
{
  require_force_completion();
}

/* ---------------------------------------------------------------------- */

void FixPIMDLangevin::complete_force_evaluation()
{
  if (method == PIMD) {
    if (mapflag)
      for (int i = 0; i < atom->nlocal; ++i) domain->unmap(atom->x[i], atom->image[i]);
    prepare_coordinates();
    spring_force();
    compute_spring_energy();
    compute_t_prim();
    compute_p_prim();
    if (mapflag)
      for (int i = 0; i < atom->nlocal; ++i) domain->unmap_inv(atom->x[i], atom->image[i]);
  } else if (method == NMPIMD) {
    inter_replica_comm(atom->f);
    nmpimd_transform((cmode == SINGLE_PROC ? bufsortedall : bufbeads), atom->f,
                     M_x2xp[universe->iworld]);
  }
  compute_pote();
  c_pe->addstep(update->ntimestep + 1);
  c_press->addstep(update->ntimestep + 1);
  force_ready = true;
}

/* ---------------------------------------------------------------------- */

void *FixPIMDLangevin::extract(const char *name, int &dim)
{
  dim = 0;
  if (strcmp(name, "pimd_bias_interface") == 0) return static_cast<PIMDBiasInterface *>(this);
  return Fix::extract(name, dim);
}

PIMDBiasCapabilities FixPIMDLangevin::bias_capabilities() const
{
  const bool supported = (strcmp(style, "pimd/langevin") == 0) && (igroup == 0) &&
      (((method == PIMD) && (ensemble == NVT)) ||
       ((method == NMPIMD) && ((ensemble == NVT) || (ensemble == NPT) || (ensemble == NPH))));
  return {np, supported, supported, supported};
}

void FixPIMDLangevin::validate_bias_order(Fix *adapter) const
{
  bool found_core = false, found_adapter = false, correct_order = false;
  int invalid = 0;
  for (const auto &fix : modify->get_fix_list()) {
    if (fix == this) found_core = true;
    if (fix == adapter) {
      correct_order = found_core;
      found_adapter = true;
    } else if (found_adapter && (modify->get_fix_mask(fix) & POST_FORCE) &&
               (strcmp(fix->style, "GROUP") != 0))
      invalid = 2;
  }
  if (!found_core || !found_adapter || !correct_order) invalid = 1;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid == 1)
    error->all(FLERR, "External PIMD bias adapter must be defined after its integrator");
  if (invalid == 2)
    error->all(FLERR,
               "External PIMD bias adapter must be defined after all other post-force fixes");
}
