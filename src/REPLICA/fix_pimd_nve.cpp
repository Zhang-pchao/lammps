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
   Shared path-integral propagation and communication.
   Derived from the PIMD implementation by Yifan Li and the FixPIMDNVE
   hierarchy developed by LiFu and Yifan Li.
------------------------------------------------------------------------- */

#include "fix_pimd_nve.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "universe.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;
using MathConst::MY_PI;
using MathConst::MY_SQRT2;
using MathConst::THIRD;
using MathSpecial::powint;

namespace {
constexpr int TAG_INTER_REPLICA_COUNT = 10;
constexpr int TAG_INTER_REPLICA_TAGS = 11;
constexpr int TAG_INTER_REPLICA_VALS = 12;
constexpr int TAG_RING_PENDING_COUNT = 400;
constexpr int TAG_RING_FOUND_COUNT = 401;
constexpr int TAG_RING_PENDING = 402;
constexpr int TAG_RING_FOUND_TAGS = 403;
constexpr int TAG_RING_FOUND_VALUES = 404;
}    // namespace

/* ---------------------------------------------------------------------- */

FixPIMDNVE::FixPIMDNVE(LAMMPS *lmp, int narg, char **arg) : FixPIMDNVE(lmp, narg, arg, true)
{
  if (domain->dimension != 3) error->all(FLERR, "Fix {} requires a 3d system", style);
  if (!atom->mass || atom->rmass_flag)
    error->all(FLERR, "Fix {} requires per-type atom masses", style);
  for (int i = 1; i < universe->nworlds; i++)
    if (universe->procs_per_world[i] != universe->procs_per_world[0])
      error->all(FLERR, "Fix {} requires the same number of processors in every partition", style);

  for (int i = 3; i < narg; i += 2) {
    if (i + 2 > narg)
      utils::missing_cmd_args(FLERR, fmt::format("fix {} {}", style, arg[i]), error);
    if (strcmp(arg[i], "method") == 0) {
      if (strcmp(arg[i + 1], "pimd") == 0)
        method = PIMD;
      else if (strcmp(arg[i + 1], "nmpimd") == 0)
        method = NMPIMD;
      else
        error->all(FLERR, "Unknown method parameter for fix {}", style);
    } else if (strcmp(arg[i], "integrator") == 0) {
      if (strcmp(arg[i + 1], "obabo") == 0)
        integrator = OBABO;
      else if (strcmp(arg[i + 1], "baoab") == 0)
        integrator = BAOAB;
      else
        error->all(FLERR, "Unknown integrator parameter for fix {}", style);
    } else if (strcmp(arg[i], "fmmode") == 0) {
      if (strcmp(arg[i + 1], "physical") == 0)
        fmmode = PHYSICAL;
      else if (strcmp(arg[i + 1], "normal") == 0)
        fmmode = NORMAL;
      else
        error->all(FLERR, "Unknown fictitious mass mode for fix {}", style);
    } else if (strcmp(arg[i], "temp") == 0) {
      temp = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (!std::isfinite(temp) || temp <= 0.0)
        error->all(FLERR, "Invalid temp value for fix {}", style);
    } else if (strcmp(arg[i], "fmass") == 0) {
      fmass = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (!std::isfinite(fmass) || fmass <= 0.0 || fmass > np)
        error->all(FLERR, "Invalid fmass value for fix {}", style);
    } else if (strcmp(arg[i], "sp") == 0) {
      sp = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (!std::isfinite(sp) || sp <= 0.0) error->all(FLERR, "Invalid sp value for fix {}", style);
    } else
      error->all(FLERR, "Unknown keyword {} for fix {}", arg[i], style);
  }
  if (method == PIMD && fmmode == NORMAL)
    error->all(FLERR, "Normal mode mass is not supported for method pimd");

  global_freq = 1;
  vector_flag = 1;
  size_vector = 10;
  extvector = -1;
  extlist = new int[size_vector];
  for (int i = 0; i < size_vector; i++) extlist[i] = (i < 7) ? 1 : 0;
  kt = force->boltz * temp;
  create_energy_computes();
  initialize_partition_communication();
}

/* ---------------------------------------------------------------------- */

FixPIMDNVE::FixPIMDNVE(LAMMPS *lmp, int narg, char **arg, bool /*defer_setup*/) :
    Fix(lmp, narg, arg), mass(nullptr), rootworld(MPI_COMM_NULL), plansend(nullptr),
    planrecv(nullptr), tagsend(nullptr), tagrecv(nullptr), bufsend(nullptr), bufrecv(nullptr),
    bufbeads(nullptr), bufsorted(nullptr), bufsortedall(nullptr), counts(nullptr),
    displacements(nullptr), lam(nullptr), M_x2xp(nullptr), M_xp2x(nullptr), M_f2fp(nullptr),
    M_fp2f(nullptr), modeindex(nullptr), _omega_k(nullptr), Lan_s(nullptr), Lan_c(nullptr),
    xc(nullptr), xcall(nullptr), x_unwrap(nullptr), id_pe(nullptr), id_press(nullptr),
    c_pe(nullptr), c_press(nullptr)
{
  time_integrate = 1;
  method = NMPIMD;
  fmmode = PHYSICAL;
  integrator = OBABO;
  np = universe->nworlds;
  inverse_np = 1.0 / np;
  normal_mode_centroid_force_scale = universe->iworld == 0 ? 1.0 / sqrt((double) np) : 0.0;
  temp = 298.15;
  hbar = kt = beta = beta_np = 0.0;
  mapflag = 1;
  removecomflag = 1;
  masstotal = 0.0;
  omega_np = fbond = spring_energy = 0.0;
  sp = fmass = 1.0;
  me = nprocs = ireplica = nreplica = nprocs_universe = 0;
  ntotal = maxlocal = maxxc = maxunwrap = maxsend = sizeplan = 0;
  x_last = x_next = 0;
  cmode = -1;
  defer_normal_mode_force = normal_mode_force_pending = 0;
  bead_bias_virial_pending = centroid_bias_virial_pending = 0;
  dtv = dtf = dtv2 = dtv3 = 0.0;
  vir = vir_ = centroid_vir = 0.0;
  t_prim = t_vir = t_cv = p_prim = p_vir = p_cv = p_md = 0.0;
  pote = tote = totke = ke_bead = se_bead = pe_bead = total_spring_energy = 0.0;
}

/* ---------------------------------------------------------------------- */

FixPIMDNVE::~FixPIMDNVE()
{
  if (id_pe) modify->delete_compute(id_pe);
  if (id_press) modify->delete_compute(id_press);
  delete[] id_pe;
  delete[] id_press;
  delete[] extlist;
  delete[] mass;
  delete[] _omega_k;
  delete[] Lan_c;
  delete[] Lan_s;
  delete[] plansend;
  delete[] planrecv;
  delete[] modeindex;
  memory->destroy(xcall);
  memory->destroy(bufsorted);
  memory->destroy(bufsortedall);
  memory->destroy(counts);
  memory->destroy(displacements);
  memory->sfree(lam);
  memory->destroy(M_x2xp);
  memory->destroy(M_xp2x);
  memory->destroy(M_f2fp);
  memory->destroy(M_fp2f);
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

int FixPIMDNVE::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::validate_partition_topology(bool require_consecutive_tags)
{
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

void FixPIMDNVE::check_atom_consistency()
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

void FixPIMDNVE::initialize_partition_communication(bool sorted_single_rank)
{
  me = comm->me;
  nprocs = comm->nprocs;
  if (sorted_single_rank && nprocs == 1)
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
  if (rootworld != MPI_COMM_NULL) MPI_Comm_free(&rootworld);
  MPI_Comm_create(universe->uworld, rootgroup, &rootworld);
  if (rootgroup != MPI_GROUP_NULL) MPI_Group_free(&rootgroup);
  if (uworldgroup != MPI_GROUP_NULL) MPI_Group_free(&uworldgroup);
  delete[] iroots;

  if (atom->natoms >= MAXSMALLINT / 3)
    error->all(FLERR, "Fix {} has too many atoms for inter-partition communication", style);
  ntotal = static_cast<int>(atom->natoms);
  if (!bufsend && !bufsorted)
    reallocate();
  else if (atom->nmax > maxlocal)
    reallocate();
  if (atom->nmax > maxunwrap) reallocate_x_unwrap();
  if (atom->nmax > maxxc) reallocate_xc();
  if (!xcall) memory->create(xcall, std::max(1, ntotal * 3), "FixPIMDNVE:xcall");
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::create_energy_computes()
{
  delete[] id_pe;
  id_pe = utils::strdup(std::string(id) + "_pimd_pe");
  modify->add_compute(std::string(id_pe) + " all pe");
  delete[] id_press;
  id_press = utils::strdup(std::string(id) + "_pimd_press");
  modify->add_compute(std::string(id_press) + " all pressure thermo_temp virial");
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::init()
{
  defer_normal_mode_force = 0;
  normal_mode_force_pending = 0;
  bead_bias_virial_pending = 0;

  validate_partition_topology(true);

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

  init_normal_mode_coefficients();
  setup_subclass_state();

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

void FixPIMDNVE::setup(int vflag)
{
  trace_nonfinite_state("setup-entry", "bead-x-v-physical-f");

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
    trace_nonfinite_state("setup-bead-to-normal-post", "normal-mode");
  } else if (method == PIMD) {
    prepare_coordinates();
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
    trace_nonfinite_state("setup-normal-to-bead-post", "bead-x-normal-vf");
  }
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) domain->unmap_inv(x[i], image[i]);
  }

  post_force(vflag);
  compute_totke();
  end_of_step();
  c_pe->addstep(update->ntimestep + 1);
  c_press->addstep(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::initial_integrate(int /*vflag*/)
{
  trace_nonfinite_state("initial-entry", "bead-x-normal-vf");
  prepare_normal_mode_forces();
  trace_nonfinite_state("initial-force-ready", "bead-x-normal-vf");

  int nlocal = atom->nlocal;
  double **x = atom->x;
  imageint *image = atom->image;
  if (mapflag) {
    for (int i = 0; i < nlocal; i++) domain->unmap(x[i], image[i]);
  }
  if (integrator == OBABO) {
    o_step();
    trace_nonfinite_state("initial-obabo-o1-post", "bead-x-normal-vf");
    b_step();
    trace_nonfinite_state("initial-obabo-b-post", "bead-x-normal-vf");
    if (method == NMPIMD) {
      inter_replica_comm(x);
      if (cmode == SINGLE_PROC)
        nmpimd_transform(bufsortedall, x, M_x2xp[universe->iworld]);
      else if (cmode == MULTI_PROC)
        nmpimd_transform(bufbeads, x, M_x2xp[universe->iworld]);
      trace_nonfinite_state("initial-obabo-bead-to-normal-post", "normal-mode");
      qc_step();
      a_step();
      qc_step();
      a_step();
      trace_nonfinite_state("initial-obabo-a-post", "normal-mode");
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
    b_step();
    trace_nonfinite_state("initial-baoab-b-post", "bead-x-normal-vf");
    if (method == NMPIMD) {
      inter_replica_comm(x);
      if (cmode == SINGLE_PROC)
        nmpimd_transform(bufsortedall, x, M_x2xp[universe->iworld]);
      else if (cmode == MULTI_PROC)
        nmpimd_transform(bufbeads, x, M_x2xp[universe->iworld]);
      trace_nonfinite_state("initial-baoab-bead-to-normal-post", "normal-mode");
      qc_step();
      a_step();
      trace_nonfinite_state("initial-baoab-a1-post", "normal-mode");
    } else if (method == PIMD) {
      q_step();
    } else {
      error->universe_all(
          FLERR,
          fmt::format("Unknown method parameter for fix {}. Only nmpimd and pimd are supported!",
                      style));
    }
    o_step();
    trace_nonfinite_state("initial-baoab-o-post", "normal-mode");
    if (method == NMPIMD) {
      qc_step();
      a_step();
      trace_nonfinite_state("initial-baoab-a2-post", "normal-mode");
    } else if (method == PIMD) {
      q_step();
      trace_nonfinite_state("initial-baoab-q2-post", "bead");
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
    trace_nonfinite_state("initial-normal-to-bead-post", "bead-x-normal-vf");
  }

  if (mapflag) {
    for (int i = 0; i < nlocal; i++) { domain->unmap_inv(x[i], image[i]); }
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::final_integrate()
{
  trace_nonfinite_state("final-entry", "bead-x-normal-v-physical-f");
  prepare_normal_mode_forces();
  trace_nonfinite_state("final-force-ready", "bead-x-normal-vf");

  b_step();
  trace_nonfinite_state("final-b-post", "bead-x-normal-vf");
  if (integrator == OBABO) {
    o_step();
    trace_nonfinite_state("final-obabo-o-post", "bead-x-normal-vf");
  } else if (integrator == BAOAB) {

  } else {
    error->universe_all(FLERR, fmt::format("Unknown integrator parameter for fix {}", style));
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::prepare_coordinates()
{ inter_replica_comm(atom->x); }

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::post_force(int /*flag*/)
{
  trace_nonfinite_state("post-physical-force", "bead-x-normal-v-physical-f");

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

  compute_vir();
  compute_xf_vir();
  compute_cvir();
  compute_t_vir();

  if (method == PIMD) {
    if (mapflag) {
      for (int i = 0; i < nlocal; i++) { domain->unmap(x[i], image[i]); }
    }
    prepare_coordinates();
    spring_force();
    compute_spring_energy();
    compute_t_prim();
    compute_p_prim();
    if (mapflag) {
      for (int i = 0; i < nlocal; i++) { domain->unmap_inv(x[i], image[i]); }
    }
  }
  if (method == NMPIMD) {
    if (defer_normal_mode_force) {
      normal_mode_force_pending = 1;
    } else {
      inter_replica_comm(f);
      if (cmode == SINGLE_PROC)
        nmpimd_transform(bufsortedall, f, M_x2xp[universe->iworld]);
      else if (cmode == MULTI_PROC)
        nmpimd_transform(bufbeads, f, M_x2xp[universe->iworld]);
      trace_nonfinite_state("post-force-transform-post", "bead-x-normal-vf");
      after_force_transform_hook();
    }
  }

  if (method == PIMD) after_force_transform_hook();
  c_pe->addstep(update->ntimestep + 1);
  c_press->addstep(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::end_of_step()
{
  if (bead_bias_virial_pending) prepare_normal_mode_forces();
  compute_pote();
  compute_totke();
  compute_p_cv();
  compute_tote();
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::collect_xc()
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

void FixPIMDNVE::b_step()
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

void FixPIMDNVE::qc_step()
{
  if (universe->iworld != 0) return;
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(atom->mask[i] & groupbit)) continue;
    for (int d = 0; d < 3; d++) atom->x[i][d] += dtv * atom->v[i][d];
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::a_step()
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

void FixPIMDNVE::q_step()
{
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(atom->mask[i] & groupbit)) continue;
    for (int d = 0; d < 3; d++) atom->x[i][d] += dtv * atom->v[i][d];
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::nmpimd_init()
{
  if (!M_x2xp) memory->create(M_x2xp, np, np, "fix_feynman:M_x2xp");
  if (!M_xp2x) memory->create(M_xp2x, np, np, "fix_feynman:M_xp2x");

  if (!lam) lam = (double *) memory->smalloc(sizeof(double) * np, "FixPIMDNVE::lam");

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

void FixPIMDNVE::init_normal_mode_coefficients()
{
  double beta = 1.0 / kt;
  const double _omega_np = np / beta / hbar;
  // The phase and velocity prefactor use the same fictitious-mass frequency.
  // Dividing by one retains the default-fmass evaluation order.
  const double _omega_np_dt_half = _omega_np * update->dt * 0.5 / sqrt(fmass);

  if (!_omega_k) _omega_k = new double[np];
  if (!Lan_c) Lan_c = new double[np];
  if (!Lan_s) Lan_s = new double[np];
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
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::nmpimd_transform(double **src, double **des, double *vector)
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

void FixPIMDNVE::prepare_normal_mode_forces()
{
  if (method != NMPIMD || !normal_mode_force_pending) return;
  trace_nonfinite_state("deferred-force-transform-pre", "bead-x-normal-v-physical-f");

  if (bead_bias_virial_pending) {
    // Pressure must include the current bias. The kinetic estimators retain
    // the physical-force values computed before PLUMED in post_force().
    compute_vir();
    compute_cvir();
    bead_bias_virial_pending = 0;
  }

  double **f = atom->f;
  inter_replica_comm(f);
  if (cmode == SINGLE_PROC)
    nmpimd_transform(bufsortedall, f, M_x2xp[universe->iworld]);
  else if (cmode == MULTI_PROC)
    nmpimd_transform(bufbeads, f, M_x2xp[universe->iworld]);
  trace_nonfinite_state("deferred-force-transform-post", "bead-x-normal-vf");
  normal_mode_force_pending = 0;
  after_force_transform_hook();
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::spring_force()
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

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::comm_init(bool neighbors_only)
{
  if (np != universe->nworlds)
    error->all(FLERR, "Fix pimd/langevin: np must equal universe->nworlds");

  int nlocal = atom->nlocal;
  if (cmode == SINGLE_PROC) {
    if (!counts) memory->create(counts, nreplica, "FixPIMDNVE:counts");
    if (!displacements) memory->create(displacements, nreplica, "FixPIMDNVE:displacements");
    for (int i = 0; i < nreplica; i++) counts[i] = 3 * nlocal;
    displacements[0] = 0;
    for (int i = 0; i < nreplica - 1; i++) displacements[i + 1] = displacements[i] + counts[i];
  }
  const int plan_size = neighbors_only ? 2 : np - 1;
  if (sizeplan != plan_size) {
    delete[] plansend;
    delete[] planrecv;
    delete[] modeindex;
    plansend = planrecv = modeindex = nullptr;
  }
  sizeplan = plan_size;
  if (!plansend) plansend = new int[sizeplan];
  if (!planrecv) planrecv = new int[sizeplan];
  if (!modeindex) modeindex = new int[sizeplan];
  for (int i = 0; i < sizeplan; i++) {
    const int step = (neighbors_only && i == 1) ? np - 1 : i + 1;

    // send to the (i+1)-th "next" replica, same local rank within that replica
    plansend[i] = universe->me + comm->nprocs * step;
    if (plansend[i] >= universe->nprocs) plansend[i] -= universe->nprocs;

    // receive from the (i+1)-th "previous" replica, same local rank within that replica
    planrecv[i] = universe->me - comm->nprocs * step;
    if (planrecv[i] < 0) planrecv[i] += universe->nprocs;

    // where to store what we receive this round:
    // this is the replica index you are pulling from in this step
    modeindex[i] = (universe->iworld + step) % universe->nworlds;
  }

  x_next = (universe->iworld + 1 + universe->nworlds) % (universe->nworlds);
  x_last = (universe->iworld - 1 + universe->nworlds) % (universe->nworlds);
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::reallocate_xc()
{
  maxxc = std::max(1, atom->nmax);
  memory->destroy(xc);
  memory->create(xc, maxxc, 3, "FixPIMDNVE:xc");
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::reallocate_x_unwrap()
{
  maxunwrap = std::max(1, atom->nmax);
  memory->destroy(x_unwrap);
  memory->create(x_unwrap, maxunwrap, 3, "FixPIMDNVE:x_unwrap");
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::reallocate()
{
  maxlocal = std::max(1, atom->nmax);
  maxsend = std::max(maxsend, maxlocal);
  ntotal = atom->natoms;
  if (cmode == SINGLE_PROC) {
    memory->destroy(bufsorted);
    memory->destroy(bufsortedall);
    memory->create(bufsorted, std::max(1, ntotal), 3, "FixPIMDNVE:bufsorted");
    memory->create(bufsortedall, std::max(1, nreplica * ntotal), 3, "FixPIMDNVE:bufsortedall");
  } else if (cmode == MULTI_PROC) {
    memory->destroy(bufsend);
    memory->destroy(bufrecv);
    memory->destroy(tagsend);
    memory->destroy(tagrecv);
    memory->destroy(bufbeads);
    memory->create(bufsend, maxsend * 3, "FixPIMDNVE:bufsend");
    memory->create(bufrecv, maxlocal * 3, "FixPIMDNVE:bufrecv");
    memory->create(tagsend, maxsend, "FixPIMDNVE:tagsend");
    memory->create(tagrecv, maxlocal, "FixPIMDNVE:tagrecv");
    memory->create(bufbeads, nreplica, maxlocal * 3, "FixPIMDNVE:bufrecv");
  }
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::inter_replica_comm(double **ptr)
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
        tagsend =
            (tagint *) memory->srealloc(tagsend, sizeof(tagint) * maxsend, "FixPIMDNVE:tagsend");
        bufsend = (double *) memory->srealloc(bufsend, sizeof(double) * 3 * maxsend,
                                              "FixPIMDNVE:bufsend");
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

void FixPIMDNVE::ring_collect(const std::vector<tagint> &requested_tags, double **source,
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

void FixPIMDNVE::remove_com_motion()
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

void FixPIMDNVE::compute_xf_vir()
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

void FixPIMDNVE::compute_cvir()
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
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_vir()
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

void FixPIMDNVE::compute_totke()
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

void FixPIMDNVE::compute_spring_energy()
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

void FixPIMDNVE::compute_pote()
{
  pe_bead = 0.0;
  pote = 0.0;
  c_pe->compute_scalar();
  pe_bead = c_pe->scalar;
  // PLUMED reports the physical bias B, while this beta/P integrator evolves
  // H_ring + P*B. Complete the energy tally without changing f_plumed or COLVAR.
  for (const auto &fix : modify->get_fix_list()) {
    int dim = -1;
    auto *physical_bias = static_cast<double *>(fix->extract("pimd_physical_bias_energy", dim));
    if (physical_bias && dim == 0)
      pe_bead += (np - (fix->thermo_energy ? 1 : 0)) * (*physical_bias);
  }
  double pot_energy_partition = pe_bead / universe->procs_per_world[universe->iworld];
  MPI_Allreduce(&pot_energy_partition, &pote, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_tote()
{ tote = totke + pote + total_spring_energy; }

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_t_prim()
{
  t_prim = 1.5 * group->count(igroup) * np * force->boltz * temp - total_spring_energy * inverse_np;
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_t_vir()
{
  t_vir = -0.5 * inverse_np * vir_;
  t_cv = 1.5 * group->count(igroup) * force->boltz * temp - 0.5 * inverse_np * centroid_vir;
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_p_prim()
{
  double inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
  p_prim = group->count(igroup) * np * force->boltz * temp * inv_volume -
      1.0 / 1.5 * inv_volume * total_spring_energy;
  p_prim *= force->nktv2p;
}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::compute_p_cv()
{
  if (centroid_bias_virial_pending) {
    compute_vir();
    centroid_bias_virial_pending = 0;
  }
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

double FixPIMDNVE::compute_vector(int n)
{
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
  return 0.0;
}

/* ---------------------------------------------------------------------- */

double **FixPIMDNVE::normal_mode_transform_buffer()
{ return cmode == SINGLE_PROC ? bufsortedall : bufbeads; }

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::o_step() {}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::setup_subclass_state() {}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::after_force_transform_hook() {}

/* ---------------------------------------------------------------------- */

void FixPIMDNVE::trace_nonfinite_state(const char * /*stage*/, const char * /*basis*/) {}
