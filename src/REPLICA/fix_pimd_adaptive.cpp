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

#include "fix_pimd_adaptive.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "pimd_adaptive_map.h"
#include "random_mars.h"
#include "universe.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>

using namespace LAMMPS_NS;

namespace {
// Inclusive wall-clock timings. No synchronization is introduced.
class AdaptivePhaseTimer {
 public:
  AdaptivePhaseTimer(bool enabled, double *record) :
      record_(enabled ? record : nullptr), start_(enabled ? MPI_Wtime() : 0.0)
  {
  }
  ~AdaptivePhaseTimer()
  {
    if (record_) {
      record_[0] += MPI_Wtime() - start_;
      record_[1] += 1.0;
    }
  }
 private:
  double *record_;
  double start_;
};
}    // namespace

FixPIMDAdaptive::FixPIMDAdaptive(LAMMPS *lmp, int narg, char **arg) :
    FixPIMDNVE(lmp, narg, arg, true)
{
  const char *profile = std::getenv("LAMMPS_PIMD_ADAPTIVE_PROFILE");
  profile_enabled = profile && (std::strcmp(profile, "1") == 0);
  const char *snapshot = std::getenv("LAMMPS_PIMD_PES_SNAPSHOT_EXPERIMENT");
  pes_snapshot_enabled = snapshot && (std::strcmp(snapshot, "1") == 0);
  const char *summary = std::getenv("LAMMPS_PIMD_FORCE_SUMMARY_EXPERIMENT");
  force_summary_enabled = summary && (std::strcmp(summary, "1") == 0);
  std::string options = pes_snapshot_enabled ? "snapshot=1;" : "snapshot=0;";
  options += force_summary_enabled ? "force_summary=1;" : "force_summary=0;";
  for (int i = 3; i < narg; ++i) options += std::to_string(std::strlen(arg[i])) + ":" + arg[i];
  int length = static_cast<int>(options.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, universe->uworld);
  std::string reference(length, '\0');
  if (universe->me == 0) reference = options;
  MPI_Bcast(reference.data(), length, MPI_CHAR, 0, universe->uworld);
  int invalid = options != reference;
  invalid |= (igroup != 0) || !PIMDMixedMap::valid(np, np);
  for (int i = 1; i < np; ++i)
    invalid |= universe->procs_per_world[i] != universe->procs_per_world[0];
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR,
               "Fix pimd/adaptive requires matching options, group all, power-of-two "
               "partitions and equal processor counts");
  bool have_temp = false, have_center = false, have_core = false, have_outer = false;
  bool have_initialization = false;
  for (int i = 3; i < narg;) {
    if (strcmp(arg[i], "center") == 0) {
      if (have_center) error->all(FLERR, "Fix pimd/adaptive accepts one center option");
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive center", error);
      if (strcmp(arg[i + 1], "group") == 0) {
        if (i + 2 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive center group", error);
        anchor_name = arg[i + 2];
        i += 3;
      } else {
        if (i + 3 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive center", error);
        for (int d = 0; d < 3; ++d) center[d] = utils::numeric(FLERR, arg[i + 1 + d], false, lmp);
        i += 4;
      }
      have_center = true;
    } else if (strcmp(arg[i], "quantum") == 0) {
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive quantum", error);
      if (!quantum_name.empty()) error->all(FLERR, "Fix pimd/adaptive accepts one quantum group");
      quantum_name = arg[i + 1];
      i += 2;
    } else if (strcmp(arg[i], "geometry") == 0) {
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive geometry", error);
      if (strcmp(arg[i + 1], "sphere") == 0) {
        region_axis = -1;
        i += 2;
      } else if (strcmp(arg[i + 1], "slab") == 0) {
        if (i + 2 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive geometry slab", error);
        const std::string axis = arg[i + 2];
        if ((axis != "x") && (axis != "y") && (axis != "z"))
          error->all(FLERR, "Fix pimd/adaptive slab axis must be x, y or z");
        region_axis = axis == "x" ? 0 : (axis == "y" ? 1 : 2);
        i += 3;
      } else
        error->all(FLERR, "Fix pimd/adaptive geometry must be sphere or slab");
    } else if (strcmp(arg[i], "initialize") == 0) {
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive initialize", error);
      if (strcmp(arg[i + 1], "auxiliary") != 0)
        error->all(FLERR, "Fix pimd/adaptive requires initialize auxiliary");
      have_initialization = true;
      i += 2;
    } else if (strcmp(arg[i], "thermostat") == 0) {
      if (i + 3 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive thermostat", error);
      if (random || (strcmp(arg[i + 1], "pile") != 0))
        error->all(FLERR, "Fix pimd/adaptive accepts one thermostat pile tau seed option");
      thermostat_tau = utils::numeric(FLERR, arg[i + 2], false, lmp);
      thermostat_seed = utils::inumeric(FLERR, arg[i + 3], false, lmp);
      if (!std::isfinite(thermostat_tau) || (thermostat_tau <= 0.0) || (thermostat_seed <= 0) ||
          (thermostat_seed > 900000000 - universe->nprocs))
        error->all(FLERR, "Fix pimd/adaptive requires a positive thermostat time and valid seed");
      random = std::make_unique<RanMars>(lmp, thermostat_seed + universe->me);
      i += 4;
    } else if ((strcmp(arg[i], "temp") == 0) || (strcmp(arg[i], "sp") == 0) ||
               (strcmp(arg[i], "core") == 0) || (strcmp(arg[i], "outer") == 0)) {
      if (i + 1 >= narg) utils::missing_cmd_args(FLERR, "fix pimd/adaptive", error);
      const double value = utils::numeric(FLERR, arg[i + 1], false, lmp);
      if (strcmp(arg[i], "temp") == 0) {
        temp = value;
        have_temp = true;
      }
      if (strcmp(arg[i], "sp") == 0) sp = value;
      if (strcmp(arg[i], "core") == 0) {
        core = value;
        have_core = true;
      }
      if (strcmp(arg[i], "outer") == 0) {
        outer = value;
        have_outer = true;
      }
      i += 2;
    } else
      error->all(FLERR, "Unknown fix pimd/adaptive keyword {}", arg[i]);
  }
  if (!have_temp || !have_center || !have_core || !have_outer || !have_initialization)
    error->all(FLERR,
               "Fix pimd/adaptive requires temp, center, core, outer and initialize auxiliary");
  if (!std::isfinite(temp) || !std::isfinite(sp) || (temp <= 0.0) || (sp <= 0.0) ||
      !std::isfinite(core) || !std::isfinite(outer) || (core < 0.0) || (outer <= core))
    error->all(FLERR,
               "Fix pimd/adaptive requires positive temperature and Planck scaling, "
               "and radii 0 <= core < outer");
  for (const double c : center)
    if (!std::isfinite(c)) error->all(FLERR, "Fix pimd/adaptive center must be finite");
  fixed_center = center;
  modes = std::make_unique<PIMDMixedMap>(np, np);
  scalar_flag = vector_flag = 1;
  size_vector = 5;
  extscalar = extvector = 1;
  global_freq = peratom_freq = nevery = 1;
  peratom_flag = restart_global = restart_peratom = 1;
  size_peratom_cols = NCOLS;
  maxexchange = NCOLS;
  array_atom = nullptr;
  grow_arrays(atom->nmax);
  atom->add_callback(this, Atom::GROW);
  atom->add_callback(this, Atom::RESTART);
  kt = force->boltz * temp;
  create_energy_computes();
  initialize_partition_communication(false);
  // Partition initialization sets the replica index used by these fixed rows.
  normal_mode_forward.resize(np);
  normal_mode_inverse.resize(np);
  for (int b = 0; b < np; ++b) {
    normal_mode_forward[b] = modes->coefficient(ireplica, b);
    normal_mode_inverse[b] = modes->coefficient(b, ireplica);
  }
  normal_mode_eigenvalue = modes->eigenvalue(ireplica);
}

FixPIMDAdaptive::~FixPIMDAdaptive()
{
  atom->delete_callback(this, Atom::GROW);
  atom->delete_callback(this, Atom::RESTART);
  memory->destroy(array_atom);
}

void FixPIMDAdaptive::init()
{
  pes_snapshot_valid = false;
  int invalid = (domain->dimension != 3) || domain->triclinic || !atom->mass || atom->rmass_flag ||
      (atom->natoms == 0) || (atom->natoms != ntotal) || (atom->map_style == Atom::MAP_NONE) ||
      !atom->tag_consecutive() || !utils::strmatch(update->integrate_style, "^verlet");
  for (int d = 0; d < 3; ++d) {
    if ((domain->boundary[d][0] >= 2) || (domain->boundary[d][1] >= 2)) invalid = 1;
    if (domain->periodicity[d] && ((region_axis < 0) || (region_axis == d)) &&
        (outer >= 0.5 * domain->prd[d]))
      invalid = 1;
  }
  const int units = strcmp(update->unit_style, "lj") == 0
      ? 1
      : (strcmp(update->unit_style, "real") == 0
             ? 2
             : (strcmp(update->unit_style, "metal") == 0 ? 3 : 0));
  int minimum, maximum;
  MPI_Allreduce(&units, &minimum, 1, MPI_INT, MPI_MIN, universe->uworld);
  MPI_Allreduce(&units, &maximum, 1, MPI_INT, MPI_MAX, universe->uworld);
  invalid |= (minimum == 0) || (minimum != maximum);
  double min_dt, max_dt;
  MPI_Allreduce(&update->dt, &min_dt, 1, MPI_DOUBLE, MPI_MIN, universe->uworld);
  MPI_Allreduce(&update->dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, universe->uworld);
  invalid |= !std::isfinite(update->dt) || (update->dt <= 0.0) || (min_dt != max_dt);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR,
               "Fix pimd/adaptive requires fixed nonempty 3d atoms, per-type masses, "
               "consecutive IDs, an atom map, matching lj/real/metal units and timesteps, "
               "Verlet, an orthorhombic cell without shrink wrap, and outer radius "
               "smaller than every periodic half-box");
  validate_partition_topology(true);
  validate_groups();
  const std::array<double, 9> cell = {domain->boxlo[0],
                                      domain->boxlo[1],
                                      domain->boxlo[2],
                                      domain->boxhi[0],
                                      domain->boxhi[1],
                                      domain->boxhi[2],
                                      static_cast<double>(domain->xperiodic),
                                      static_cast<double>(domain->yperiodic),
                                      static_cast<double>(domain->zperiodic)};
  invalid = (initialized || restored) && (cell != fixed_cell);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/adaptive requires an unchanged simulation cell");
  fixed_cell = cell;
  comm_init();
  hbar = sp * force->hplanck / MathConst::MY_2PI;
  dtf = 0.5 * update->dt * force->ftm2v;
  c_pe = modify->get_compute_by_id(id_pe);
  if (!c_pe) error->all(FLERR, "Fix pimd/adaptive potential energy compute is missing");
  if (!initialized && !restored) {
    for (int i = 0; i < atom->nlocal; ++i) {
      domain->unmap(atom->x[i], atom->image[i], array_atom[i]);
      for (int d = 0; d < 3; ++d) array_atom[i][3 + d] = atom->v[i][d];
    }
  }
  invalid = 0;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 6; ++d) invalid |= !std::isfinite(array_atom[i][d]);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/adaptive auxiliary state must be finite");
  std::vector<double> previous;
  if (initialized || restored) {
    previous.resize(6 * atom->nlocal);
    for (int i = 0; i < atom->nlocal; ++i) {
      domain->unmap(atom->x[i], atom->image[i], previous.data() + 6 * i);
      for (int d = 0; d < 3; ++d) previous[6 * i + 3 + d] = atom->v[i][d];
    }
  }
  map_positions();
  map_velocities();
  invalid = 0;
  if (initialized || restored) {
    for (int i = 0; i < atom->nlocal; ++i) {
      double current[6];
      domain->unmap(atom->x[i], atom->image[i], current);
      for (int d = 0; d < 3; ++d) current[3 + d] = atom->v[i][d];
      for (int d = 0; d < 6; ++d)
        if (std::abs(current[d] - previous[6 * i + d]) >
            1.0e-10 * std::max(1.0, std::abs(current[d])))
          invalid = 1;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(
        FLERR,
        "Fix pimd/adaptive physical coordinates or velocities were changed outside integration");
  initialized = true;
  c_pe->addstep(update->ntimestep);
}

void FixPIMDAdaptive::validate_companions()
{
  int invalid = 0;
  for (auto *fix : modify->get_fix_list()) {
    if ((fix == this) || (bias_registration && (fix == bias_registration->adapter))) continue;
    if ((strcmp(fix->style, "ave/time") != 0) && (strcmp(fix->style, "print") != 0)) invalid = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid)
    error->all(FLERR,
               "Fix pimd/adaptive supports only print, ave/time and one registered bias as "
               "companion fixes");
  if (bias_registration) validate_bias_order(bias_registration->adapter);
}

void FixPIMDAdaptive::inter_replica_comm(double **ptr)
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data());
  if (nprocs_universe != nreplica) {
    FixPIMDNVE::inter_replica_comm(ptr);
    return;
  }

  // One rank owns every atom of each bead. Tags remain canonical even when
  // local atom order differs between beads after sorting.
  if (!bufsend || (atom->nmax > maxlocal)) reallocate();
  const std::size_t count = 3 * static_cast<std::size_t>(ntotal);
  collective_send.resize(std::max<std::size_t>(1, count));
  collective_values.resize(std::max<std::size_t>(1, count * nreplica));
  for (int i = 0; i < atom->nlocal; ++i) {
    const std::size_t slot = 3 * static_cast<std::size_t>(atom->tag[i] - 1);
    for (int d = 0; d < 3; ++d) collective_send[slot + d] = ptr[i][d];
  }
  // rootworld is ordered by replica index, including reordered universes.
  MPI_Allgather(collective_send.data(), static_cast<int>(count), MPI_DOUBLE,
                collective_values.data(), static_cast<int>(count), MPI_DOUBLE, rootworld);
  for (int b = 0; b < nreplica; ++b)
    for (int i = 0; i < atom->nlocal; ++i) {
      const std::size_t slot = count * b + 3 * static_cast<std::size_t>(atom->tag[i] - 1);
      for (int d = 0; d < 3; ++d) bufbeads[b][3 * i + d] = collective_values[slot + d];
    }
}

void FixPIMDAdaptive::collect(int offset)
{
  rows.resize(std::max(1, atom->nlocal), array_atom[0] + offset);
  for (int i = 0; i < atom->nlocal; ++i) rows[i] = array_atom[i] + offset;
  inter_replica_comm(rows.data());
}

void FixPIMDAdaptive::validate_groups()
{
  for (int selection = 0; selection < 2; ++selection) {
    const std::string &name = selection == 0 ? anchor_name : quantum_name;
    int &bit = selection == 0 ? anchor_bit : quantum_bit;
    auto &previous = selection == 0 ? anchor_members : quantum_members;
    if (name.empty()) continue;
    const int index = group->find(name);
    int invalid = index < 0;
    if (!invalid) invalid = group->dynamic[index];
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
    if (invalid)
      error->all(FLERR,
                 "Fix pimd/adaptive reference and quantum groups must be existing static groups");
    bit = group->bitmask[index];
    std::vector<int> members(ntotal, 0);
    for (int i = 0; i < atom->nlocal; ++i)
      if (atom->mask[i] & bit) members[atom->tag[i] - 1] = 1;
    MPI_Allreduce(MPI_IN_PLACE, members.data(), ntotal, MPI_INT, MPI_SUM, world);
    auto minimum = members, maximum = members;
    MPI_Allreduce(MPI_IN_PLACE, minimum.data(), ntotal, MPI_INT, MPI_MIN, universe->uworld);
    MPI_Allreduce(MPI_IN_PLACE, maximum.data(), ntotal, MPI_INT, MPI_MAX, universe->uworld);
    if ((minimum != maximum) || (std::count(members.begin(), members.end(), 1) == 0) ||
        (!previous.empty() && (members != previous)))
      error->all(FLERR,
                 "Fix pimd/adaptive reference and quantum groups must be nonempty and unchanged in "
                 "all partitions");
    previous = std::move(members);
  }
}

void FixPIMDAdaptive::update_anchor(const std::vector<double> &state, bool velocity)
{
  if (anchor_name.empty()) return;
  double reference[3] = {};
  if (!velocity) {
    const tagint first =
        std::find(anchor_members.begin(), anchor_members.end(), 1) - anchor_members.begin() + 1;
    for (int i = 0; i < atom->nlocal; ++i)
      if (atom->tag[i] == first)
        std::copy(state.data() + 3 * i, state.data() + 3 * i + 3, reference);
    MPI_Allreduce(MPI_IN_PLACE, reference, 3, MPI_DOUBLE, MPI_SUM, world);
  }
  double sums[4] = {}, low[3] = {}, high[3] = {};
  for (int i = 0; i < atom->nlocal; ++i) {
    if (!(atom->mask[i] & anchor_bit)) continue;
    const double mass = atom->mass[atom->type[i]];
    double delta[3];
    for (int d = 0; d < 3; ++d) delta[d] = state[3 * i + d] - reference[d];
    if (!velocity) domain->minimum_image(FLERR, delta[0], delta[1], delta[2]);
    for (int d = 0; d < 3; ++d) {
      sums[d] += mass * delta[d];
      low[d] = std::min(low[d], delta[d]);
      high[d] = std::max(high[d], delta[d]);
    }
    sums[3] += mass;
  }
  MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_DOUBLE, MPI_SUM, world);
  if (!std::isfinite(sums[3]) || (sums[3] <= 0.0))
    error->all(FLERR, "Fix pimd/adaptive center group requires positive finite total mass");
  anchor_mass = sums[3];
  if (!velocity) {
    MPI_Allreduce(MPI_IN_PLACE, low, 3, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(MPI_IN_PLACE, high, 3, MPI_DOUBLE, MPI_MAX, world);
    for (int d = 0; d < 3; ++d)
      if (domain->periodicity[d] && (high[d] - low[d] >= 0.5 * domain->prd[d]))
        error->all(FLERR, "Fix pimd/adaptive center group must fit within each periodic half-box");
  }
  for (int d = 0; d < 3; ++d) {
    const double value = reference[d] + sums[d] / sums[3];
    if (velocity)
      reference_velocity[d] = value;
    else
      center[d] = value;
  }
}

void FixPIMDAdaptive::map_positions()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 4);
  collect(0);
  if (!xc || (atom->nmax > maxxc)) reallocate_xc();
  centroids.resize(std::max(1, 3 * atom->nlocal));
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      xc[i][d] = 0.0;
      for (int b = 0; b < np; ++b) xc[i][d] += inverse_np * bufbeads[b][3 * i + d];
      centroids[3 * i + d] = xc[i][d];
    }
  update_anchor(centroids, false);
  for (int i = 0; i < atom->nlocal; ++i) {
    double delta[3];
    for (int d = 0; d < 3; ++d) delta[d] = xc[i][d] - center[d];
    domain->minimum_image(FLERR, delta[0], delta[1], delta[2]);
    double a = PIMDAdaptiveMap::amplitude(delta, core, outer, array_atom[i] + 10, region_axis);
    if (atom->mask[i] & quantum_bit) {
      a = 1.0;
      std::fill(array_atom[i] + 10, array_atom[i] + 13, 0.0);
    }
    array_atom[i][9] = a;
    for (int d = 0; d < 3; ++d) atom->x[i][d] = xc[i][d] + a * (array_atom[i][d] - xc[i][d]);
    domain->unmap_inv(atom->x[i], atom->image[i]);
  }
  capture_pes_snapshot();
}

void FixPIMDAdaptive::map_velocities(bool positions_just_mapped)
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 6);
  // Only initial_integrate uses the immediately preceding map_positions result.
  // Other callers recollect after possible migration before using local indices.
  if (!positions_just_mapped) collect(0);
  offsets.resize(std::max(1, 3 * atom->nlocal));
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      double c = positions_just_mapped ? xc[i][d] : 0.0;
      if (!positions_just_mapped)
        for (int b = 0; b < np; ++b) c += inverse_np * bufbeads[b][3 * i + d];
      offsets[3 * i + d] = array_atom[i][d] - c;
    }
  collect(3);
  mean_velocity.assign(std::max(1, 3 * atom->nlocal), 0.0);
  auto &mean = mean_velocity;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d)
      for (int b = 0; b < np; ++b) mean[3 * i + d] += inverse_np * bufbeads[b][3 * i + d];
  update_anchor(mean, true);
  for (int i = 0; i < atom->nlocal; ++i) {
    double rate = 0.0;
    for (int d = 0; d < 3; ++d)
      rate += array_atom[i][10 + d] * (mean[3 * i + d] - reference_velocity[d]);
    for (int d = 0; d < 3; ++d)
      atom->v[i][d] = PIMDAdaptiveMap::velocity(array_atom[i][9], array_atom[i][3 + d],
                                                mean[3 * i + d], offsets[3 * i + d], rate);
  }
}

void FixPIMDAdaptive::transform(int offset, bool forward)
{
  collect(offset);
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 2);
  const auto &coefficients = forward ? normal_mode_forward : normal_mode_inverse;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      double value = 0.0;
      for (int b = 0; b < np; ++b)
        value += coefficients[b] * bufbeads[b][3 * i + d];
      array_atom[i][offset + d] = value;
    }
}

void FixPIMDAdaptive::kick()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 18);
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d)
      array_atom[i][3 + d] += dtf / atom->mass[atom->type[i]] * array_atom[i][6 + d];
}

void FixPIMDAdaptive::propagate(double dt)
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 20);
  const double omega = np * kt / hbar * std::sqrt(normal_mode_eigenvalue);
  const double c = std::cos(omega * dt), s = std::sin(omega * dt);
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      const double x = array_atom[i][d], v = array_atom[i][3 + d];
      if (ireplica == 0)
        array_atom[i][d] = x + dt * v;
      else {
        array_atom[i][d] = c * x + s / omega * v;
        array_atom[i][3 + d] = c * v - omega * s * x;
      }
    }
}

void FixPIMDAdaptive::o_step()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 22);
  if (!random) return;
  const double omega = np * kt / hbar * std::sqrt(normal_mode_eigenvalue);
  const double gamma = ireplica == 0 ? 1.0 / thermostat_tau : 2.0 * omega;
  const double damping = std::exp(-gamma * update->dt);
  for (int i = 0; i < atom->nlocal; ++i) {
    const double noise = std::sqrt(-std::expm1(-2.0 * gamma * update->dt) * np * kt /
                                   (force->mvv2e * atom->mass[atom->type[i]]));
    for (int d = 0; d < 3; ++d)
      array_atom[i][3 + d] = damping * array_atom[i][3 + d] + noise * random->gaussian();
  }
}

void FixPIMDAdaptive::initial_integrate(int /*vflag*/)
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 10);
  require_force_completion();
  kick();
  transform(0, true);
  transform(3, true);
  propagate(0.5 * update->dt);
  o_step();
  propagate(0.5 * update->dt);
  transform(0, false);
  transform(3, false);
  map_positions();
  map_velocities(true);
}

void FixPIMDAdaptive::final_integrate()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 12);
  require_force_completion();
  kick();
  map_velocities();
}

void FixPIMDAdaptive::pullback()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 8);
  collect(0);
  offsets.resize(std::max(1, np * 3 * atom->nlocal));
  double spring = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const double factor =
        0.5 * force->mvv2e * atom->mass[atom->type[i]] * np * (kt / hbar) * (kt / hbar);
    for (int d = 0; d < 3; ++d) {
      double c = 0.0;
      for (int b = 0; b < np; ++b) c += inverse_np * bufbeads[b][3 * i + d];
      for (int b = 0; b < np; ++b) offsets[(i * np + b) * 3 + d] = bufbeads[b][3 * i + d] - c;
      const double delta = array_atom[i][d] - bufbeads[(ireplica + 1) % np][3 * i + d];
      spring += factor * delta * delta;
    }
  }
  MPI_Allreduce(&spring, &values[1], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  // The single-rank-per-bead route needs only the force mean and its
  // contraction with the auxiliary offset. Keep the general gather fallback.
  const bool reduced = force_summary_enabled && (nprocs_universe == nreplica) &&
      (ntotal <= std::numeric_limits<int>::max() / 4);
  int invalid = 0;
  if (reduced) {
    const std::size_t count = 4 * static_cast<std::size_t>(ntotal);
    force_summary.resize(std::max<std::size_t>(1, count));
    for (int i = 0; i < atom->nlocal; ++i) {
      const std::size_t slot = 4 * static_cast<std::size_t>(atom->tag[i] - 1);
      double moment = 0.0;
      for (int d = 0; d < 3; ++d) {
        const double f = atom->f[i][d];
        invalid |= !std::isfinite(f);
        force_summary[slot + d] = inverse_np * f;
        moment += inverse_np * f * offsets[(i * np + ireplica) * 3 + d];
      }
      force_summary[slot + 3] = moment;
    }
    MPI_Allreduce(MPI_IN_PLACE, force_summary.data(), static_cast<int>(count),
                  MPI_DOUBLE, MPI_SUM, rootworld);
    ++force_summary_counts[0];
  } else {
    inter_replica_comm(atom->f);
    ++force_summary_counts[1];
  }
  double reaction[3] = {};
  for (int i = 0; i < atom->nlocal; ++i) {
    double mean[3] = {}, moment = 0.0;
    if (reduced) {
      const std::size_t slot = 4 * static_cast<std::size_t>(atom->tag[i] - 1);
      for (int d = 0; d < 3; ++d) mean[d] = force_summary[slot + d];
      moment = force_summary[slot + 3];
    } else {
      for (int b = 0; b < np; ++b)
        for (int d = 0; d < 3; ++d) {
          const double f = bufbeads[b][3 * i + d];
          invalid |= !std::isfinite(f);
          mean[d] += inverse_np * f;
          moment += inverse_np * f * offsets[(i * np + b) * 3 + d];
        }
    }
    for (int d = 0; d < 3; ++d) {
      array_atom[i][6 + d] = PIMDAdaptiveMap::force(array_atom[i][9], atom->f[i][d], mean[d],
                                                    array_atom[i][10 + d], moment);
      reaction[d] -= array_atom[i][10 + d] * moment;
    }
  }
  if (!anchor_name.empty()) {
    MPI_Allreduce(MPI_IN_PLACE, reaction, 3, MPI_DOUBLE, MPI_SUM, world);
    for (int i = 0; i < atom->nlocal; ++i)
      if (atom->mask[i] & anchor_bit)
        for (int d = 0; d < 3; ++d)
          array_atom[i][6 + d] += atom->mass[atom->type[i]] / anchor_mass * reaction[d];
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "Fix pimd/adaptive received non-finite physical forces");
}

void FixPIMDAdaptive::setup(int vflag)
{
  post_force(vflag);
  if (!bias_pending) end_of_step();
}

void FixPIMDAdaptive::post_force(int /*vflag*/)
{
  validate_companions();
  if (bias_pending) error->all(FLERR, "Missing external PIMD bias force submission");
  ++bias_generation;
  force_ready = false;
  bias_pending = bool(bias_registration);
  bias_energy = 0.0;
  if (bias_pending) {
    for (double &value : bias_registration->adapter->virial) value = 0.0;
    if (!x_unwrap || (atom->nmax > maxunwrap)) reallocate_x_unwrap();
    for (int i = 0; i < atom->nlocal; ++i) domain->unmap(atom->x[i], atom->image[i], x_unwrap[i]);
    inter_replica_comm(x_unwrap);
    std::fill(xcall, xcall + 3 * ntotal, 0.0);
    for (int i = 0; i < atom->nlocal; ++i)
      for (int d = 0; d < 3; ++d)
        for (int b = 0; b < np; ++b)
          xcall[3 * (atom->tag[i] - 1) + d] += inverse_np * bufbeads[b][3 * i + d];
    MPI_Allreduce(MPI_IN_PLACE, xcall, 3 * ntotal, MPI_DOUBLE, MPI_SUM, world);
  } else {
    pullback();
    force_ready = true;
  }
  c_pe->addstep(update->ntimestep + 1);
}

void FixPIMDAdaptive::end_of_step()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 14);
  require_force_completion();
  double physical = c_pe->compute_scalar();
  if (bias_registration && bias_registration->adapter->energy_global_flag &&
      bias_registration->adapter->thermo_energy)
    physical -= bias_energy;
  const double potential = physical / (np * static_cast<double>(comm->nprocs));
  const double bias_share = bias_energy / static_cast<double>(comm->nprocs);
  MPI_Allreduce(&bias_share, &values[4], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  MPI_Allreduce(&potential, &values[0], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  double kinetic = 0.0;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d)
      kinetic += 0.5 * force->mvv2e * atom->mass[atom->type[i]] * inverse_np *
          array_atom[i][3 + d] * array_atom[i][3 + d];
  MPI_Allreduce(&kinetic, &values[2], 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  values[3] = values[0] + values[1] + values[2] + values[4];
}

double FixPIMDAdaptive::compute_scalar()
{ return values[3]; }
double FixPIMDAdaptive::compute_vector(int n)
{ return values[n]; }
PIMDBiasCapabilities FixPIMDAdaptive::bias_capabilities() const
{ return {np, true, true, true}; }

void *FixPIMDAdaptive::extract(const char *name, int &dim)
{
  dim = 0;
  if (strcmp(name, "pimd_pes_view_v1") == 0) {
    refresh_pes_view();
    return &pes_view;
  }
  if (strcmp(name, "pimd_pes_snapshot_counts") == 0) {
    dim = 1;
    return pes_snapshot_counts.data();
  }
  if (strcmp(name, "pimd_force_summary_counts") == 0) {
    dim = 1;
    return force_summary_counts.data();
  }
  if (strcmp(name, "pimd_adaptive_profile") == 0) {
    dim = 1;
    return phase_profile.data();
  }
  if (strcmp(name, "np") == 0) return &np;
  return FixPIMDNVE::extract(name, dim);
}

void FixPIMDAdaptive::capture_pes_snapshot()
{
  pes_snapshot_valid = false;
  if (!pes_snapshot_enabled || (comm->nprocs != 1) || (atom->nlocal != ntotal)) return;
  pes_snapshot_auxiliary.resize(3 * static_cast<std::size_t>(ntotal));
  pes_snapshot_centroid.resize(pes_snapshot_auxiliary.size());
  pes_snapshot_seen.assign(ntotal, 0);
  for (int i = 0; i < atom->nlocal; ++i) {
    const tagint tag = atom->tag[i];
    if ((tag < 1) || (tag > ntotal) || pes_snapshot_seen[tag - 1]) return;
    pes_snapshot_seen[tag - 1] = 1;
    const std::size_t slot = 3 * static_cast<std::size_t>(tag - 1);
    for (int d = 0; d < 3; ++d) {
      pes_snapshot_auxiliary[slot + d] = array_atom[i][d];
      pes_snapshot_centroid[slot + d] = centroids[3 * i + d];
    }
  }
  pes_snapshot_valid = true;
}

bool FixPIMDAdaptive::pes_snapshot_matches()
{
  if (!pes_snapshot_valid || (comm->nprocs != 1) || (atom->nlocal != ntotal) ||
      (atom->natoms != ntotal)) return false;
  std::fill(pes_snapshot_seen.begin(), pes_snapshot_seen.end(), 0);
  for (int i = 0; i < atom->nlocal; ++i) {
    const tagint tag = atom->tag[i];
    if ((tag < 1) || (tag > ntotal) || pes_snapshot_seen[tag - 1]) return false;
    pes_snapshot_seen[tag - 1] = 1;
    const std::size_t slot = 3 * static_cast<std::size_t>(tag - 1);
    for (int d = 0; d < 3; ++d)
      if (array_atom[i][d] != pes_snapshot_auxiliary[slot + d]) return false;
  }
  return true;
}

void FixPIMDAdaptive::refresh_pes_view()
{
  AdaptivePhaseTimer phase_timer(profile_enabled, phase_profile.data() + 16);
  // Combine freshness with the existing collective initialization guard. A
  // mutation on any bead forces every rank onto the original gather path.
  int status[2] = {!initialized, !(pes_snapshot_enabled && pes_snapshot_matches())};
  MPI_Allreduce(MPI_IN_PLACE, status, 2, MPI_INT, MPI_MAX, universe->uworld);
  if (status[0]) error->all(FLERR, "PIMD potential input requires initialized auxiliary paths");
  const bool reuse = status[1] == 0;
  if (!reuse) collect(0);
  ++pes_snapshot_counts[reuse ? 0 : 1];
  int invalid = 0;
  pes_view.beads = np;
  pes_view.bead = ireplica;
  pes_view.nlocal = atom->nlocal;
  pes_view.natoms = atom->natoms;
  ++pes_view.generation;
  pes_view.tags.resize(atom->nlocal);
  pes_view.centroid.assign(3 * atom->nlocal, 0.0);
  pes_view.amplitude.resize(atom->nlocal);
  for (int i = 0; i < atom->nlocal; ++i) {
    pes_view.tags[i] = atom->tag[i];
    pes_view.amplitude[i] = array_atom[i][9];
    invalid |= !std::isfinite(pes_view.amplitude[i]) || (pes_view.amplitude[i] < 0.0) ||
        (pes_view.amplitude[i] > 1.0);
    for (int d = 0; d < 3; ++d) {
      if (reuse)
        pes_view.centroid[3 * i + d] =
            pes_snapshot_centroid[3 * static_cast<std::size_t>(atom->tag[i] - 1) + d];
      else
        for (int b = 0; b < np; ++b)
          pes_view.centroid[3 * i + d] += inverse_np * bufbeads[b][3 * i + d];
      invalid |= !std::isfinite(pes_view.centroid[3 * i + d]);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, universe->uworld);
  if (invalid) error->all(FLERR, "PIMD potential input contains invalid path data");
}

void FixPIMDAdaptive::submit_bias(const PIMDBiasRegistration &registration,
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

  // Physical bias gradients are normalized by the adapter. Multiply by P
  // before applying the same auxiliary pullback as the physical PES.
  collect(0);
  offsets.resize(std::max(1, 3 * np * atom->nlocal));
  bias_centers.resize(std::max(1, 3 * atom->nlocal));
  auto &centers = bias_centers;
  for (int i = 0; i < atom->nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      double c = 0.0;
      for (int b = 0; b < np; ++b) c += inverse_np * bufbeads[b][3 * i + d];
      centers[3 * i + d] = c;
      for (int b = 0; b < np; ++b) offsets[(i * np + b) * 3 + d] = bufbeads[b][3 * i + d] - c;
    }
  bias_forces.resize(std::max(1, 3 * atom->nlocal));
  bias_force_rows.resize(std::max(1, atom->nlocal), bias_forces.data());
  auto &forces = bias_forces;
  auto &force_rows = bias_force_rows;
  // resize may move the data; refresh even the zero-local-rank sentinel.
  force_rows[0] = forces.data();
  for (int i = 0; i < atom->nlocal; ++i) {
    force_rows[i] = forces.data() + 3 * i;
    for (int d = 0; d < 3; ++d) force_rows[i][d] = submission.force[3 * i + d];
  }
  inter_replica_comm(force_rows.data());
  for (int k = 0; k < 6; ++k) dynamical_virial[k] = np * submission.virial[k];
  constexpr int row[6] = {0, 1, 2, 0, 0, 1}, column[6] = {0, 1, 2, 1, 2, 2};
  for (int i = 0; i < atom->nlocal; ++i) {
    double moment = 0.0;
    for (int b = 0; b < np; ++b)
      for (int d = 0; d < 3; ++d) moment += bufbeads[b][3 * i + d] * offsets[(i * np + b) * 3 + d];
    double distance[3];
    for (int d = 0; d < 3; ++d) distance[d] = centers[3 * i + d] - center[d];
    domain->minimum_image(FLERR, distance[0], distance[1], distance[2]);
    // Anchor is fixed in Cartesian space under this strain derivative.
    // d(c-H*n)/d(strain) supplies distance+center, not just distance.
    for (int k = 0; k < 6; ++k)
      dynamical_virial[k] += (distance[row[k]] + (anchor_name.empty() ? center[row[k]] : 0.0)) *
          array_atom[i][10 + column[k]] * moment;
    for (int d = 0; d < 3; ++d) atom->f[i][d] += np * submission.force[3 * i + d];
  }
  bias_energy = submission.energy;
  c_pe->invoked_scalar = -1;
  bias_pending = false;
  pullback();
  force_ready = true;
  end_of_step();
}

void FixPIMDAdaptive::grow_arrays(int nmax)
{
  const int old = capacity;
  capacity = std::max(1, nmax);
  memory->grow(array_atom, capacity, NCOLS, "pimd/adaptive:state");
  for (int i = old; i < capacity; ++i) std::fill(array_atom[i], array_atom[i] + NCOLS, 0.0);
}
void FixPIMDAdaptive::copy_arrays(int i, int j, int /*delflag*/)
{ std::copy(array_atom[i], array_atom[i] + NCOLS, array_atom[j]); }
int FixPIMDAdaptive::pack_exchange(int i, double *buf)
{
  std::copy(array_atom[i], array_atom[i] + NCOLS, buf);
  return NCOLS;
}
int FixPIMDAdaptive::unpack_exchange(int i, double *buf)
{
  pes_snapshot_valid = false;
  std::copy(buf, buf + NCOLS, array_atom[i]);
  return NCOLS;
}
int FixPIMDAdaptive::pack_restart(int i, double *buf)
{
  buf[0] = NCOLS + 1;
  pack_exchange(i, buf + 1);
  return NCOLS + 1;
}
void FixPIMDAdaptive::unpack_restart(int i, int nth)
{
  int m = 0;
  for (int k = 0; k < nth; ++k) m += static_cast<int>(atom->extra[i][m]);
  if (atom->extra[i][m] != NCOLS + 1)
    error->one(FLERR, "Invalid fix pimd/adaptive per-atom restart size");
  unpack_exchange(i, atom->extra[i] + m + 1);
}
int FixPIMDAdaptive::maxsize_restart()
{ return NCOLS + 1; }
int FixPIMDAdaptive::size_restart(int /*i*/)
{ return NCOLS + 1; }
double FixPIMDAdaptive::memory_usage()
{
  return static_cast<double>(capacity) * NCOLS * sizeof(double) +
      (offsets.capacity() + centroids.capacity() + mean_velocity.capacity() +
       bias_centers.capacity() + bias_forces.capacity() + normal_mode_forward.capacity() +
       normal_mode_inverse.capacity() + collective_send.capacity() + collective_values.capacity() +
       force_summary.capacity()) *
      sizeof(double) +
      (rows.capacity() + bias_force_rows.capacity()) * sizeof(double *) +
      (pes_snapshot_auxiliary.capacity() + pes_snapshot_centroid.capacity()) * sizeof(double) +
      pes_snapshot_seen.capacity() * sizeof(unsigned char);
}

void FixPIMDAdaptive::write_restart(FILE *fp)
{
  if (!initialized) error->all(FLERR, "Fix pimd/adaptive must run before writing a restart");
  const std::size_t header_size = 29 + anchor_members.size() + quantum_members.size();
  const std::size_t count =
      header_size + (random ? static_cast<std::size_t>(comm->nprocs) * RanMars::STATE_SIZE : 0);
  if (count > MAXSMALLINT / sizeof(double))
    error->all(FLERR, "Fix pimd/adaptive restart contains too many processors");
  std::vector<double> data(count);
  const double header[] = {81754.0,
                           3.0,
                           static_cast<double>(np),
                           static_cast<double>(ntotal),
                           temp,
                           sp,
                           random ? 1.0 : 0.0,
                           thermostat_tau,
                           static_cast<double>(thermostat_seed),
                           static_cast<double>(comm->nprocs),
                           fixed_center[0],
                           fixed_center[1],
                           fixed_center[2],
                           core,
                           outer};
  std::copy(header, header + 15, data.begin());
  std::copy(fixed_cell.begin(), fixed_cell.end(), data.begin() + 15);
  data[24] = region_axis;
  data[25] = anchor_name.empty() ? 0.0 : 1.0;
  data[26] = anchor_members.size();
  data[27] = quantum_name.empty() ? 0.0 : 1.0;
  data[28] = quantum_members.size();
  std::copy(anchor_members.begin(), anchor_members.end(), data.begin() + 29);
  std::copy(quantum_members.begin(), quantum_members.end(),
            data.begin() + 29 + anchor_members.size());
  if (random) {
    double state[RanMars::STATE_SIZE];
    random->get_state(state);
    MPI_Gather(state, RanMars::STATE_SIZE, MPI_DOUBLE, data.data() + header_size,
               RanMars::STATE_SIZE, MPI_DOUBLE, 0, world);
  }
  if (comm->me == 0) {
    const int bytes = static_cast<int>(data.size() * sizeof(double));
    fwrite(&bytes, sizeof(int), 1, fp);
    fwrite(data.data(), sizeof(double), data.size(), fp);
  }
}

void FixPIMDAdaptive::restart(char * /*buf*/)
{ error->all(FLERR, "Fix pimd/adaptive requires size-checked restart data"); }

void FixPIMDAdaptive::restart(char *buf, int nbytes)
{
  pes_snapshot_valid = false;
  int header_size = 24;
  if ((nbytes < header_size * static_cast<int>(sizeof(double))) || (nbytes % sizeof(double) != 0))
    error->all(FLERR, "Invalid fix pimd/adaptive restart size");
  const auto *data = reinterpret_cast<const double *>(buf);
  if ((data[0] != 81754.0) || ((data[1] != 1.0) && (data[1] != 2.0) && (data[1] != 3.0)) ||
      (data[2] != np) || (data[3] != ntotal) || (data[4] != temp) || (data[5] != sp) ||
      (data[6] != (random ? 1.0 : 0.0)) || (data[7] != thermostat_tau) ||
      (data[8] != thermostat_seed) || (data[10] != fixed_center[0]) ||
      (data[11] != fixed_center[1]) || (data[12] != fixed_center[2]) || (data[13] != core) ||
      (data[14] != outer))
    error->all(FLERR, "Fix pimd/adaptive restart settings do not match");
  if (data[1] == 1.0) {
    if ((region_axis != -1) || !anchor_name.empty() || !quantum_name.empty())
      error->all(FLERR, "Fix pimd/adaptive restart region settings do not match");
  } else {
    const int base = data[1] == 3.0 ? 29 : 27;
    if (nbytes < base * static_cast<int>(sizeof(double)))
      error->all(FLERR, "Invalid fix pimd/adaptive region restart size");
    const int members = anchor_name.empty() ? 0 : ntotal;
    const int protected_members = quantum_name.empty() ? 0 : ntotal;
    if ((data[24] != region_axis) || (data[25] != (anchor_name.empty() ? 0.0 : 1.0)) ||
        (data[26] != members) ||
        (members > MAXSMALLINT / static_cast<int>(sizeof(double)) - base) ||
        (protected_members > MAXSMALLINT / static_cast<int>(sizeof(double)) - base - members))
      error->all(FLERR, "Fix pimd/adaptive restart region settings do not match");
    if ((base == 27 && protected_members != 0) ||
        (base == 29 &&
         ((data[27] != (quantum_name.empty() ? 0.0 : 1.0)) || (data[28] != protected_members))))
      error->all(FLERR, "Fix pimd/adaptive restart quantum group settings do not match");
    header_size = base + members + protected_members;
    if (nbytes < header_size * static_cast<int>(sizeof(double)))
      error->all(FLERR, "Invalid fix pimd/adaptive region restart size");
    anchor_members.resize(members);
    quantum_members.resize(protected_members);
    for (int i = 0; i < members + protected_members; ++i) {
      if ((data[base + i] != 0.0) && (data[base + i] != 1.0))
        error->all(FLERR, "Invalid fix pimd/adaptive restart group membership");
      if (i < members)
        anchor_members[i] = static_cast<int>(data[base + i]);
      else
        quantum_members[i - members] = static_cast<int>(data[base + i]);
    }
  }
  if (!std::isfinite(data[9]) || (data[9] < 1.0) || (data[9] > MAXSMALLINT) ||
      (data[9] != std::floor(data[9])) || (random && (data[9] != comm->nprocs)))
    error->all(FLERR, "Fix pimd/adaptive PILE restart requires the same processor count");
  const std::size_t count =
      header_size + (random ? static_cast<std::size_t>(comm->nprocs) * RanMars::STATE_SIZE : 0);
  if (nbytes / sizeof(double) != count) error->all(FLERR, "Invalid fix pimd/adaptive restart size");
  std::copy(data + 15, data + 24, fixed_cell.begin());
  if (random) {
    for (int rank = 0; rank < comm->nprocs; ++rank) {
      const double *state = data + header_size + rank * RanMars::STATE_SIZE;
      int invalid = state[0] != -1.0;
      for (int k = 0; k < RanMars::STATE_SIZE; ++k) invalid |= !std::isfinite(state[k]);
      for (int k = 1; k < 98; ++k) invalid |= (state[k] < 0.0) || (state[k] >= 1.0);
      for (int k : {98, 99})
        invalid |= (state[k] < 1.0) || (state[k] > 97.0) || (state[k] != std::floor(state[k]));
      for (int k : {100, 101, 102}) invalid |= (state[k] < 0.0) || (state[k] >= 1.0);
      invalid |= (state[103] != 0.0) && (state[103] != 1.0);
      if (invalid) error->all(FLERR, "Invalid fix pimd/adaptive restart random state");
    }
    double state[RanMars::STATE_SIZE];
    std::copy(data + header_size + comm->me * RanMars::STATE_SIZE,
              data + header_size + (comm->me + 1) * RanMars::STATE_SIZE, state);
    random->set_state(state);
  }
  restored = true;
}
