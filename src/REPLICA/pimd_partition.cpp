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

#include "pimd_partition.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "lammps.h"
#include "lmptype.h"
#include "universe.h"

#include <algorithm>
#include <vector>

using namespace LAMMPS_NS;

namespace {
constexpr int TAG_RING_PENDING_COUNT = 400;
constexpr int TAG_RING_FOUND_COUNT = 401;
constexpr int TAG_RING_PENDING = 402;
constexpr int TAG_RING_FOUND_TAGS = 403;
constexpr int TAG_RING_FOUND_VALUES = 404;
}    // namespace

/* ----------------------------------------------------------------------
   check per-atom type and group consistency across all partitions
------------------------------------------------------------------------- */

void PIMDUtils::check_atom_consistency(LAMMPS *lmp, const char *style, int groupbit)
{
  Atom *atom = lmp->atom;
  Comm *comm = lmp->comm;
  Universe *universe = lmp->universe;

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

/* ----------------------------------------------------------------------
   collect requested owned-atom vectors within one bead world
------------------------------------------------------------------------- */

void PIMDUtils::collect_atom_vectors(LAMMPS *lmp, const char *style,
                                     const std::vector<tagint> &requested_tags, double **source,
                                     std::vector<double> &requested_values)
{
  Atom *atom = lmp->atom;
  Comm *comm = lmp->comm;
  Universe *universe = lmp->universe;

  const int me = comm->me;
  const int nprocs = comm->nprocs;
  const int next = (me + 1) % nprocs;
  const int prev = (me - 1 + nprocs) % nprocs;
  const int nlocal = atom->nlocal;

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

    MPI_Sendrecv(pending.data(), send_pending, MPI_LMP_TAGINT, next, TAG_RING_PENDING,
                 incoming_pending.data(), recv_pending, MPI_LMP_TAGINT, prev, TAG_RING_PENDING,
                 lmp->world, MPI_STATUS_IGNORE);
    MPI_Sendrecv(found_tags.data(), send_found, MPI_LMP_TAGINT, next, TAG_RING_FOUND_TAGS,
                 incoming_found_tags.data(), recv_found, MPI_LMP_TAGINT, prev, TAG_RING_FOUND_TAGS,
                 lmp->world, MPI_STATUS_IGNORE);
    MPI_Sendrecv(found_values.data(), 3 * send_found, MPI_DOUBLE, next, TAG_RING_FOUND_VALUES,
                 incoming_found_values.data(), 3 * recv_found, MPI_DOUBLE, prev,
                 TAG_RING_FOUND_VALUES, lmp->world, MPI_STATUS_IGNORE);

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
