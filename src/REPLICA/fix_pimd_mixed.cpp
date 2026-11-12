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

#include "fix_pimd_mixed.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "universe.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

FixPIMDMixed::FixPIMDMixed(LAMMPS *lmp, int narg, char **arg) : FixPIMDNVE(lmp, narg, arg, true)
{
  // Validate identical options before parsing can fail in only one partition.
  std::string options;
  for (int i = 3; i < narg; ++i) options += std::to_string(std::strlen(arg[i])) + ":" + arg[i];
  int length = static_cast<int>(options.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, universe->uworld);
  std::string reference(length, '\0');
  if (universe->me == 0) reference = options;
  MPI_Bcast(reference.data(), length, MPI_CHAR, 0, universe->uworld);
  int invalid = options != reference;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/mixed options must match in every partition");
  if (!PIMDMixedMap::valid(np, np))
    error->all(FLERR, "Fix pimd/mixed requires a power-of-two partition count");
  invalid = igroup != 0;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/mixed requires group all");
  for (int i = 1; i < np; ++i)
    if (universe->procs_per_world[i] != universe->procs_per_world[0])
      error->all(FLERR, "Fix pimd/mixed requires equal processor counts in all partitions");

  bool have_temp = false;
  for (int i = 3; i < narg;) {
    if (strcmp(arg[i], "beads") == 0) {
      if (i + 2 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/mixed beads", error);
      const int p = utils::inumeric(FLERR, arg[i + 2], false, lmp);
      if (!PIMDMixedMap::valid(np, p))
        error->all(FLERR,
                   "Fix pimd/mixed bead counts must be powers of two within the partition count");
      bead_groups.emplace_back(arg[i + 1]);
      group_beads.push_back(p);
      i += 3;
    } else if (strcmp(arg[i], "thermostat") == 0) {
      if (i + 3 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/mixed thermostat", error);
      if (mixed_random || (strcmp(arg[i + 1], "pile") != 0))
        error->all(FLERR, "Fix pimd/mixed accepts one thermostat pile tau seed option");
      thermostat_tau = utils::numeric(FLERR, arg[i + 2], false, lmp);
      thermostat_seed = utils::inumeric(FLERR, arg[i + 3], false, lmp);
      if (!std::isfinite(thermostat_tau) || (thermostat_tau <= 0.0) || (thermostat_seed <= 0) ||
          (thermostat_seed > 900000000 - universe->nprocs))
        error->all(FLERR,
                   "Fix pimd/mixed requires a positive thermostat time and a seed "
                   "small enough for all processors");
      mixed_random = std::make_unique<RanMars>(lmp, thermostat_seed + universe->me);
      i += 4;
    } else if ((strcmp(arg[i], "temp") == 0) || (strcmp(arg[i], "sp") == 0)) {
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/mixed", error);
      const double value = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (!std::isfinite(value) || (value <= 0.0))
        error->all(FLERR, "Fix pimd/mixed temperature and Planck scaling must be positive");
      if (strcmp(arg[i], "temp") == 0) {
        temp = value;
        have_temp = true;
      } else
        sp = value;
      i += 2;
    } else
      error->all(FLERR, "Unknown fix pimd/mixed keyword {}", arg[i]);
  }
  if (!have_temp) error->all(FLERR, "Fix pimd/mixed requires a spring temperature");
  scalar_flag = vector_flag = 1;
  size_vector = 7;
  extscalar = extvector = 1;
  global_freq = 1;
  peratom_flag = 1;
  size_peratom_cols = 3;
  peratom_freq = 1;
  array_atom = nullptr;
  restart_global = 1;
  nevery = 1;
  kt = force->boltz * temp;
  create_energy_computes();
  initialize_partition_communication(false);
}

FixPIMDMixed::~FixPIMDMixed()
{ memory->destroy(array_atom); }

const PIMDMixedMap &FixPIMDMixed::atom_map(int i) const
{ return maps.at(bead_counts[atom->tag[i] - 1]); }

void FixPIMDMixed::initialize_counts()
{
  std::vector<int> counts(ntotal, 0);
  std::vector<int> bits;
  int invalid = 0;
  for (const auto &name : bead_groups) {
    const int index = group->find(name);
    if ((index < 0) || ((index >= 0) && group->dynamic[index])) invalid = 1;
    bits.push_back(index < 0 ? 0 : group->bitmask[index]);
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/mixed bead groups must exist and be static");
  for (int i = 0; i < atom->nlocal; ++i) {
    int selected = 0, count = 1;
    for (std::size_t g = 0; g < bits.size(); ++g)
      if (atom->mask[i] & bits[g]) {
        ++selected;
        count = group_beads[g];
      }
    if (selected > 1) invalid = 1;
    counts[atom->tag[i] - 1] = count;
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/mixed bead groups must not overlap");
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), ntotal, MPI_INT, MPI_SUM, world);
  std::vector<int> reference(counts);
  MPI_Bcast(reference.data(), ntotal, MPI_INT, 0, universe->uworld);
  invalid = counts != reference;
  if (initialized && (counts != bead_counts)) invalid = 1;
  if (!saved_counts.empty() && (counts != saved_counts)) invalid = 1;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR, "Fix pimd/mixed bead assignment must match across partitions and restarts");
  if (*std::max_element(counts.begin(), counts.end()) != np)
    error->all(FLERR, "Fix pimd/mixed maximum bead count must equal the partition count");
  bead_counts = std::move(counts);
  unique_beads = 0;
  for (const int p : bead_counts) {
    unique_beads += p;
    if (!maps.count(p)) maps.emplace(p, PIMDMixedMap(np, p));
  }
}

void FixPIMDMixed::init()
{
  int invalid = (domain->dimension != 3) || !atom->mass || atom->rmass_flag ||
      (atom->natoms == 0) || (atom->natoms != ntotal);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR, "Fix pimd/mixed requires fixed nonempty 3d atoms with per-type masses");
  const int units = strcmp(update->unit_style, "lj") == 0
      ? 1
      : (strcmp(update->unit_style, "real") == 0
             ? 2
             : (strcmp(update->unit_style, "metal") == 0 ? 3 : 0));
  int min_units, max_units;
  MPI_Allreduce(&units, &min_units, 1, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(&units, &max_units, 1, MPI_INT, MPI_MAX, universe->uworld);
  if ((min_units == 0) || (min_units != max_units))
    error->all(FLERR, "Fix pimd/mixed requires matching lj, real, or metal units");
  double min_dt, max_dt;
  MPI_Allreduce(&update->dt, &min_dt, 1, MPI_DOUBLE, MPI_MIN, universe->uworld);
  MPI_Allreduce(&update->dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, universe->uworld);
  invalid = !std::isfinite(update->dt) || (update->dt <= 0.0) || (min_dt != max_dt);
  if (!utils::strmatch(update->integrate_style, "^verlet")) invalid = 1;
  if ((atom->map_style == Atom::MAP_NONE) || !atom->tag_consecutive()) invalid = 1;
  for (int d = 0; d < 3; ++d)
    if ((domain->boundary[d][0] >= 2) || (domain->boundary[d][1] >= 2)) invalid = 1;
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR,
               "Fix pimd/mixed requires matching positive timesteps, Verlet integration, "
               "an atom map, consecutive atom IDs, and no shrink-wrap boundaries");
  validate_partition_topology(true);
  const std::array<double, 9> cell = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2],
                                      domain->boxhi[0], domain->boxhi[1], domain->boxhi[2],
                                      domain->xy,       domain->xz,       domain->yz};
  invalid = initialized && (cell != fixed_cell);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/mixed requires an unchanged simulation cell");
  fixed_cell = cell;
  initialize_counts();
  comm_init();
  hbar = sp * force->hplanck / MathConst::MY_2PI;
  dtf = 0.5 * update->dt * force->ftm2v;
  c_pe = modify->get_compute_by_id(id_pe);
  if (!c_pe) error->all(FLERR, "Fix pimd/mixed potential energy compute is missing");
  validate_copies();
  initialized = true;
  c_pe->addstep(update->ntimestep);
}

// Registration occurs during the adapter's init(), after the integrator's init().
// Validate all companions once every adapter has initialized, before using forces.
void FixPIMDMixed::validate_companions()
{
  int invalid = 0;
  for (auto *fix : modify->get_fix_list()) {
    if ((fix == this) || (bias_registration && (fix == bias_registration->adapter))) continue;
    if ((strcmp(fix->style, "GROUP") != 0) && (strcmp(fix->style, "ave/time") != 0) &&
        (strcmp(fix->style, "print") != 0))
      invalid = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(
        FLERR,
        "Fix pimd/mixed supports only print, ave/time and one registered bias as companion fixes");
  if (bias_registration) validate_bias_order(bias_registration->adapter);
}

void FixPIMDMixed::unwrap_coordinates()
{
  if (!x_unwrap || (atom->nmax > maxunwrap)) reallocate_x_unwrap();
  for (int i = 0; i < atom->nlocal; ++i) domain->unmap(atom->x[i], atom->image[i], x_unwrap[i]);
}

void FixPIMDMixed::validate_copies()
{
  unwrap_coordinates();
  int invalid = 0;
  for (double **ptr : {x_unwrap, atom->v}) {
    inter_replica_comm(ptr);
    for (int i = 0; i < atom->nlocal; ++i) {
      const auto &map = atom_map(i);
      const int owner = map.representative(map.bead_index(ireplica));
      for (int d = 0; d < 3; ++d) {
        const double value = bufbeads[owner][3 * i + d];
        if (!std::isfinite(ptr[i][d]) || !std::isfinite(value) ||
            (std::abs(ptr[i][d] - value) > 1.0e-12 * std::max(1.0, std::abs(value))))
          invalid = 1;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR,
               "Fix pimd/mixed copies of each independent bead must have matching positions and "
               "velocities");
}

void FixPIMDMixed::setup(int vflag)
{
  post_force(vflag);
  if (!bias_pending) end_of_step();
}

void FixPIMDMixed::kick()
{
  for (int i = 0; i < atom->nlocal; ++i) {
    if (!atom_map(i).is_representative(ireplica)) continue;
    const double factor = dtf / atom->mass[atom->type[i]];
    for (int d = 0; d < 3; ++d) atom->v[i][d] += factor * atom->f[i][d];
  }
}

void FixPIMDMixed::transform(double **ptr, bool forward)
{
  inter_replica_comm(ptr);
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    if (forward && !map.is_representative(ireplica)) continue;
    const int target = map.bead_index(ireplica);
    for (int d = 0; d < 3; ++d) {
      double value = 0.0;
      for (int k = 0; k < map.beads(); ++k) {
        const double coefficient = forward ? map.coefficient(target, k)
                                           : map.coefficient(k, target);
        value += coefficient * bufbeads[map.representative(k)][3 * i + d];
      }
      ptr[i][d] = value;
    }
  }
}

void FixPIMDMixed::propagate_modes(double dt)
{
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    if (!map.is_representative(ireplica)) continue;
    const int mode = map.bead_index(ireplica);
    const double omega = map.beads() * kt / hbar * std::sqrt(map.eigenvalue(mode));
    const double c = std::cos(omega * dt);
    const double s = std::sin(omega * dt);
    for (int d = 0; d < 3; ++d) {
      const double x = atom->x[i][d], v = atom->v[i][d];
      if (mode == 0)
        atom->x[i][d] = x + dt * v;
      else {
        atom->x[i][d] = c * x + s / omega * v;
        atom->v[i][d] = -omega * s * x + c * v;
      }
    }
  }
}

void FixPIMDMixed::o_step()
{
  if (!mixed_random) return;
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    if (!map.is_representative(ireplica)) continue;
    const int mode = map.bead_index(ireplica);
    const double omega = map.beads() * kt / hbar * std::sqrt(map.eigenvalue(mode));
    const double gamma = mode == 0 ? 1.0 / thermostat_tau : 2.0 * omega;
    const double damping = std::exp(-gamma * update->dt);
    const double noise = std::sqrt(-std::expm1(-2.0 * gamma * update->dt) * map.beads() * kt /
                                   (force->mvv2e * atom->mass[atom->type[i]]));
    for (int d = 0; d < 3; ++d)
      atom->v[i][d] = damping * atom->v[i][d] + noise * mixed_random->gaussian();
  }
}

void FixPIMDMixed::initial_integrate(int /*vflag*/)
{
  require_force_completion();
  kick();
  for (int i = 0; i < atom->nlocal; ++i) domain->unmap(atom->x[i], atom->image[i]);
  transform(atom->x, true);
  transform(atom->v, true);
  propagate_modes(0.5 * update->dt);
  o_step();
  propagate_modes(0.5 * update->dt);
  transform(atom->x, false);
  transform(atom->v, false);
  for (int i = 0; i < atom->nlocal; ++i) domain->unmap_inv(atom->x[i], atom->image[i]);
}

void FixPIMDMixed::replicate_velocities()
{
  inter_replica_comm(atom->v);
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    const int owner = map.representative(map.bead_index(ireplica));
    for (int d = 0; d < 3; ++d) atom->v[i][d] = bufbeads[owner][3 * i + d];
  }
}

void FixPIMDMixed::final_integrate()
{
  require_force_completion();
  kick();
  replicate_velocities();
}

void FixPIMDMixed::update_metadata()
{
  if (!array_atom || (atom->nmax > metadata_capacity)) {
    metadata_capacity = std::max(1, atom->nmax);
    memory->destroy(array_atom);
    memory->create(array_atom, metadata_capacity, 3, "pimd/mixed:metadata");
  }
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    array_atom[i][0] = map.beads();
    array_atom[i][1] = map.bead_index(ireplica) + 1;
    array_atom[i][2] = map.is_representative(ireplica) ? 1.0 : 0.0;
  }
}

void FixPIMDMixed::post_force(int /*vflag*/)
{
  validate_companions();
  if (bias_pending) error->all(FLERR, "Missing external PIMD bias force submission");
  ++bias_generation;
  force_ready = false;
  bias_pending = bool(bias_registration);
  bias_energy = 0.0;
  if (bias_registration)
    for (double &value : bias_registration->adapter->virial) value = 0.0;
  unwrap_coordinates();
  if (!xc || (atom->nmax > maxxc)) reallocate_xc();
  inter_replica_comm(x_unwrap);
  double spring = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    for (int d = 0; d < 3; ++d) {
      xc[i][d] = 0.0;
      for (int a = 0; a < map.beads(); ++a)
        xc[i][d] += bufbeads[map.representative(a)][3 * i + d] / map.beads();
    }
    if (!map.is_representative(ireplica)) continue;
    const int next = map.representative((map.bead_index(ireplica) + 1) % map.beads());
    const double factor =
        0.5 * force->mvv2e * atom->mass[atom->type[i]] * map.beads() * (kt / hbar) * (kt / hbar);
    for (int d = 0; d < 3; ++d) {
      const double delta = x_unwrap[i][d] - bufbeads[next][3 * i + d];
      spring += factor * delta * delta;
    }
  }
  if (bias_pending) {
    std::fill(xcall, xcall + 3 * ntotal, 0.0);
    for (int i = 0; i < atom->nlocal; ++i)
      for (int d = 0; d < 3; ++d) xcall[3 * (atom->tag[i] - 1) + d] = xc[i][d];
    MPI_Allreduce(MPI_IN_PLACE, xcall, 3 * ntotal, MPI_DOUBLE, MPI_SUM, world);
  }
  MPI_Allreduce(&spring, &values[1], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  inter_replica_comm(atom->f);
  double cvir = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    for (int d = 0; d < 3; ++d) {
      const double physical =
          map.physical_force(bufbeads[0] + 3 * i + d, 3 * maxlocal, map.bead_index(ireplica));
      if (map.is_representative(ireplica)) cvir += (x_unwrap[i][d] - xc[i][d]) * physical;
      atom->f[i][d] = map.beads() * physical;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &cvir, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  values[4] = 1.5 * unique_beads * kt - values[1];
  values[5] = 1.5 * ntotal * kt - 0.5 * cvir;
  update_metadata();
  force_ready = !bias_pending;
  c_pe->addstep(update->ntimestep + 1);
}

void FixPIMDMixed::end_of_step()
{
  require_force_completion();
  double physical = c_pe->compute_scalar();
  if (bias_registration) {
    const auto *adapter = bias_registration->adapter;
    if (adapter->energy_global_flag && adapter->thermo_energy) physical -= bias_energy;
  }
  const double potential = physical / (np * static_cast<double>(comm->nprocs));
  const double bias_share = bias_energy / static_cast<double>(comm->nprocs);
  MPI_Allreduce(&bias_share, &values[6], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  MPI_Allreduce(&potential, &values[0], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  double kinetic = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    if (!map.is_representative(ireplica)) continue;
    const double factor = 0.5 * force->mvv2e * atom->mass[atom->type[i]] / map.beads();
    for (int d = 0; d < 3; ++d) kinetic += factor * atom->v[i][d] * atom->v[i][d];
  }
  MPI_Allreduce(&kinetic, &values[2], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  values[3] = values[0] + values[1] + values[2] + values[6];
}

double FixPIMDMixed::compute_scalar()
{ return values[3]; }
double FixPIMDMixed::compute_vector(int n)
{ return values[n]; }

void *FixPIMDMixed::extract(const char *name, int &dim)
{
  dim = 0;
  if (strcmp(name, "np") == 0) return &np;
  return FixPIMDNVE::extract(name, dim);
}

PIMDBiasCapabilities FixPIMDMixed::bias_capabilities() const
{ return {np, true, true, false}; }

void FixPIMDMixed::submit_bias(const PIMDBiasRegistration &registration,
                               const PIMDBiasSubmission &submission, double *dynamical_virial)
{
  int state_error = !bias_pending ? 1 : 0;
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

  // Reuse tag-aware transport without mixing physical forces and adapter buffers.
  // The map divides by P; the physical bias gradient is already normalized.
  std::vector<double> contribution(std::max(1, 3 * atom->nlocal));
  std::vector<double *> rows(std::max(1, atom->nlocal), contribution.data());
  for (int i = 0; i < atom->nlocal; ++i) {
    rows[i] = contribution.data() + 3 * i;
    for (int d = 0; d < 3; ++d) rows[i][d] = submission.force[3 * i + d];
  }
  inter_replica_comm(rows.data());
  for (int i = 0; i < atom->nlocal; ++i) {
    const auto &map = atom_map(i);
    for (int d = 0; d < 3; ++d)
      atom->f[i][d] += np * static_cast<double>(map.beads()) *
          map.physical_force(bufbeads[0] + 3 * i + d, 3 * maxlocal, map.bead_index(ireplica));
  }
  bias_energy = submission.energy;
  for (int d = 0; d < 6; ++d) dynamical_virial[d] = np * submission.virial[d];
  c_pe->invoked_scalar = -1;
  bias_pending = false;
  force_ready = true;
  end_of_step();
}

void FixPIMDMixed::write_restart(FILE *fp)
{
  if (!initialized) error->all(FLERR, "Fix pimd/mixed must run before writing a restart");
  const std::size_t count = 10 + static_cast<std::size_t>(ntotal) +
      (mixed_random ? static_cast<std::size_t>(comm->nprocs) * RanMars::STATE_SIZE : 0);
  if (count > MAXSMALLINT / sizeof(double))
    error->all(FLERR, "Fix pimd/mixed restart contains too many atoms or processors");
  std::vector<double> data(count);
  const double header[] = {81753.0,
                           2.0,
                           static_cast<double>(np),
                           static_cast<double>(ntotal),
                           temp,
                           sp,
                           mixed_random ? 1.0 : 0.0,
                           thermostat_tau,
                           static_cast<double>(thermostat_seed),
                           static_cast<double>(comm->nprocs)};
  std::copy(header, header + 10, data.begin());
  for (int i = 0; i < ntotal; ++i) data[10 + i] = bead_counts[i];
  if (mixed_random) {
    double state[RanMars::STATE_SIZE];
    mixed_random->get_state(state);
    MPI_Gather(state, RanMars::STATE_SIZE, MPI_DOUBLE, data.data() + 10 + ntotal,
               RanMars::STATE_SIZE, MPI_DOUBLE, 0, world);
  }
  if (comm->me == 0) {
    const int bytes = static_cast<int>(data.size() * sizeof(double));
    fwrite(&bytes, sizeof(int), 1, fp);
    fwrite(data.data(), sizeof(double), data.size(), fp);
  }
}

void FixPIMDMixed::restart(char * /*buf*/)
{ error->all(FLERR, "Fix pimd/mixed requires size-checked restart data"); }

void FixPIMDMixed::restart(char *buf, int nbytes)
{
  if ((nbytes < 10 * static_cast<int>(sizeof(double))) || (nbytes % sizeof(double) != 0))
    error->all(FLERR, "Invalid fix pimd/mixed restart size");
  const auto *data = reinterpret_cast<const double *>(buf);
  if ((data[0] != 81753.0) || (data[1] != 2.0) || (data[2] != np) || (data[3] != ntotal) ||
      (data[4] != temp) || (data[5] != sp) || (data[6] != (mixed_random ? 1.0 : 0.0)) ||
      (data[7] != thermostat_tau) || (data[8] != thermostat_seed))
    error->all(FLERR, "Fix pimd/mixed restart settings do not match");
  if (!std::isfinite(data[9]) || (data[9] < 1.0) || (data[9] > MAXSMALLINT) ||
      (data[9] != std::floor(data[9])))
    error->all(FLERR, "Invalid fix pimd/mixed restart processor count");
  if (mixed_random && (data[9] != comm->nprocs))
    error->all(FLERR, "Fix pimd/mixed PILE restart requires the same processor count");
  const std::size_t count = 10 + static_cast<std::size_t>(ntotal) +
      (mixed_random ? static_cast<std::size_t>(comm->nprocs) * RanMars::STATE_SIZE : 0);
  if (nbytes / sizeof(double) != count) error->all(FLERR, "Invalid fix pimd/mixed restart size");
  saved_counts.resize(ntotal);
  for (int i = 0; i < ntotal; ++i) {
    const double p = data[i + 10];
    if (!std::isfinite(p) || (p < 1.0) || (p > np) || (p != std::floor(p)) ||
        !PIMDMixedMap::valid(np, static_cast<int>(p)))
      error->all(FLERR, "Invalid fix pimd/mixed restart bead count");
    saved_counts[i] = static_cast<int>(p);
  }
  if (mixed_random) {
    const double *saved = data + 10 + ntotal;
    // Validate every stored rank before selecting this rank's complete stream.
    for (int rank = 0; rank < comm->nprocs; ++rank) {
      const double *state = saved + rank * RanMars::STATE_SIZE;
      int invalid = state[0] != -1.0;
      for (int k = 0; k < RanMars::STATE_SIZE; ++k)
        if (!std::isfinite(state[k])) invalid = 1;
      for (int k = 1; k < 98; ++k)
        if ((state[k] < 0.0) || (state[k] >= 1.0)) invalid = 1;
      for (int k : {98, 99})
        if ((state[k] < 1.0) || (state[k] > 97.0) || (state[k] != std::floor(state[k])))
          invalid = 1;
      for (int k : {100, 101, 102})
        if ((state[k] < 0.0) || (state[k] >= 1.0)) invalid = 1;
      if ((state[103] != 0.0) && (state[103] != 1.0)) invalid = 1;
      if (invalid) error->all(FLERR, "Invalid fix pimd/mixed restart random state");
    }
    double state[RanMars::STATE_SIZE];
    std::copy(saved + comm->me * RanMars::STATE_SIZE, saved + (comm->me + 1) * RanMars::STATE_SIZE,
              state);
    mixed_random->set_state(state);
  }
}
