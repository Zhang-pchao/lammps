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

#define LAMMPS_LIB_MPI 1
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "fix.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "universe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;

namespace {
LAMMPS *open_system(int partitions)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string layout = std::to_string(partitions) + "x" + std::to_string(size / partitions);
    const char *args[]       = {"LAMMPS_test", "-screen",      "none", "-log", "none",
                                "-partition",  layout.c_str(), "-in",  "none", "-nocite"};
    return static_cast<LAMMPS *>(
        lammps_open(10, const_cast<char **>(args), MPI_COMM_WORLD, nullptr));
}

bool commands(LAMMPS *lmp, const std::string &input)
{
    lammps_commands_string(lmp, input.c_str());
    int bad            = lammps_has_error(lmp);
    char message[1024] = {};
    if (bad) lammps_get_last_error_message(lmp, message, sizeof(message));
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(bad, 0) << message;
    return bad == 0;
}

int bead_count(int tag, int partitions, int mode)
{
    if (partitions == 1 || tag > 3) return mode == 2 ? partitions : 1;
    if (mode == 2 || tag != 1) return partitions;
    return mode == 1 ? 2 : 1;
}

std::string fix_command(int partitions, int mode, bool pile)
{
    std::string result = "fix pi all pimd/mixed temp 300 ";
    if (partitions == 1 || mode == 2)
        result += "beads all " + std::to_string(partitions);
    else {
        result += "beads fine " + std::to_string(partitions);
        if (mode == 1) result += " beads coarse 2";
    }
    if (pile) result += " thermostat pile 100 19771";
    return result + "\n";
}

// Flexible original-TIP3P parameters from doc/src/Howto_tip3p.rst, with
// cutoff Coulomb and no constraints. This is a molecular engineering fixture.
bool prepare(LAMMPS *lmp, int partitions, int mode = 1, bool pile = false, int molecules = 2,
             bool native_nve = false, bool newton = true, bool continuous_energy = false)
{
    const int b       = lmp->universe->iworld;
    std::string input = "units real\natom_style full\natom_modify map array sort 0 0\n";
    if (lmp->comm->nprocs == 2) input += "processors 2 1 1\n";
    if (!newton) input += "newton off\n";
    // Cutoff Coulomb has an energy discontinuity. The NVE convergence fixture
    // contains the complete finite cluster inside the larger interaction range.
    input += continuous_energy ? "region box block 0 24 0 24 0 24\n"
                               : "region box block 0 12 0 12 0 12\n";
    input += "create_box 2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 "
             "extra/angle/per/atom 1 extra/special/per/atom 2\n";
    for (int mol = 0; mol < molecules; ++mol) {
        const double ox = 4.5 + 3.1 * (mol % 2), oy = 4.5 + 3.1 * (mol / 2);
        const double base[3][3] = {
            {ox, oy, 5.0}, {ox + 0.75695, oy + 0.58588, 5.0}, {ox - 0.75695, oy + 0.58588, 5.0}};
        for (int a = 0; a < 3; ++a) {
            const int p        = bead_count(3 * mol + a + 1, partitions, mode);
            const int index    = b / (partitions / p);
            const double delta = 0.002 * (index - 0.5 * (p - 1));
            input += "create_atoms " + std::to_string(a ? 2 : 1) + " single " +
                     std::to_string(base[a][0] + delta) + " " + std::to_string(base[a][1]) + " " +
                     std::to_string(base[a][2] - 0.5 * delta) + "\n";
        }
    }
    input += "mass 1 15.9994\nmass 2 1.008\nset type 1 charge -0.834\nset type 2 charge 0.417\n";
    input += continuous_energy ? "pair_style lj/cut/coul/cut 10\n"
                               : "pair_style lj/cut/coul/cut 5\n";
    input += "pair_coeff 1 1 0.1521 3.1507\npair_coeff 2 2 0 1\n"
             "pair_modify mix arithmetic\nbond_style harmonic\nbond_coeff 1 450 0.9572\n"
             "angle_style harmonic\nangle_coeff 1 55 104.52\nspecial_bonds lj/coul 0 0 0\n";
    for (int mol = 0; mol < molecules; ++mol) {
        const std::string o = std::to_string(3 * mol + 1), h1 = std::to_string(3 * mol + 2);
        const std::string h2 = std::to_string(3 * mol + 3);
        input += "create_bonds single/bond 1 " + o + " " + h1 + "\n";
        input += "create_bonds single/bond 1 " + o + " " + h2 + "\n";
        input += "create_bonds single/angle 1 " + h1 + " " + o + " " + h2 + "\n";
        for (int a = 1; a <= 3; ++a)
            input += "set atom " + std::to_string(3 * mol + a) + " mol " + std::to_string(mol + 1) +
                     "\n";
    }
    input += "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
             "group fine id 2 3\ngroup coarse id 1\nvelocity all set 0.001 0.002 -0.001\n"
             "timestep 0.05\nthermo 100000\n";
    input += native_nve ? "fix pi all nve\n" : fix_command(partitions, mode, pile);
    return commands(lmp, input + "run 0");
}

std::vector<double> state(LAMMPS *lmp)
{
    const int n = lmp->atom->natoms;
    std::vector<double> values(9 * n * lmp->universe->nworlds, 0.0);
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int start = 9 * (n * lmp->universe->iworld + lmp->atom->tag[i] - 1);
        double x[3];
        lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], x);
        for (int d = 0; d < 3; ++d) {
            values[start + d]     = x[d];
            values[start + 3 + d] = lmp->atom->v[i][d];
            values[start + 6 + d] = lmp->atom->f[i][d];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return values;
}

void check_copies_and_geometry(LAMMPS *lmp, int partitions, int mode)
{
    const auto values = state(lmp);
    const int n       = lmp->atom->natoms;
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](double v) {
        return std::isfinite(v);
    }));
    for (int atom = 0; atom < n; ++atom) {
        const int stride = partitions / bead_count(atom + 1, partitions, mode);
        for (int b = 0; b < partitions; ++b)
            for (int d = 0; d < 9; ++d)
                EXPECT_DOUBLE_EQ(values[9 * (n * b + atom) + d],
                                 values[9 * (n * (b / stride * stride) + atom) + d]);
    }
    for (int b = 0; b < partitions; ++b)
        for (int mol = 0; mol < n / 3; ++mol)
            for (int h = 1; h <= 2; ++h) {
                double r2 = 0;
                for (int d = 0; d < 3; ++d) {
                    double delta =
                        values[9 * (n * b + 3 * mol + h) + d] - values[9 * (n * b + 3 * mol) + d];
                    const double length = lmp->domain->boxhi[d] - lmp->domain->boxlo[d];
                    delta -= length * std::nearbyint(delta / length);
                    r2 += delta * delta;
                }
                EXPECT_GT(std::sqrt(r2), 0.5);
                EXPECT_LT(std::sqrt(r2), 1.5);
            }
    auto *fix = lmp->modify->get_fix_by_id("pi");
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int p = bead_count(lmp->atom->tag[i], partitions, mode), b = lmp->universe->iworld;
        EXPECT_DOUBLE_EQ(fix->array_atom[i][0], p);
        EXPECT_DOUBLE_EQ(fix->array_atom[i][1], b / (partitions / p) + 1);
        EXPECT_DOUBLE_EQ(fix->array_atom[i][2], b % (partitions / p) == 0 ? 1 : 0);
    }
}

std::vector<double> momentum(LAMMPS *lmp, int partitions, int mode)
{
    std::vector<double> result(3, 0);
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int p = bead_count(lmp->atom->tag[i], partitions, mode);
        if (lmp->universe->iworld % (partitions / p)) continue;
        for (int d = 0; d < 3; ++d)
            result[d] += lmp->atom->mass[lmp->atom->type[i]] / p * lmp->atom->v[i][d];
    }
    MPI_Allreduce(MPI_IN_PLACE, result.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

void restart_roundtrip(bool pile)
{
    std::vector<double> reference, resumed;
    auto *continuous = open_system(4);
    if (prepare(continuous, 4, 1, pile) && commands(continuous, "run 100"))
        reference = state(continuous);
    lammps_close(continuous);
    auto *first            = open_system(4);
    const std::string file = "mixed_water.restart." + std::to_string(first->universe->iworld);
    if (prepare(first, 4, 1, pile) && commands(first, "run 51"))
        commands(first, "write_restart " + file);
    lammps_close(first);
    auto *second = open_system(4);
    if (commands(second, "atom_modify map array sort 0 0\nprocessors 2 1 1\nread_restart " + file +
                             "\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n" +
                             fix_command(4, 1, pile) + "run 49")) {
        resumed = state(second);
        check_copies_and_geometry(second, 4, 1);
    }
    if (second->comm->me == 0) std::remove(file.c_str());
    lammps_close(second);
    ASSERT_EQ(reference.size(), 216);
    ASSERT_EQ(resumed.size(), reference.size());
    for (std::size_t k = 0; k < reference.size(); ++k)
        EXPECT_NEAR(reference[k], resumed[k], 1.0e-9);
}
} // namespace

TEST(PIMDMixedMolecular, every_unique_force_matches_physical_energy_difference)
{
    for (bool newton : {true, false}) {
        auto *lmp = open_system(4);
        if (prepare(lmp, 4, 1, false, 2, false, newton)) {
            const auto initial = state(lmp);
            auto *fix          = lmp->modify->get_fix_by_id("pi");
            for (int tag = 1; tag <= 6; ++tag) {
                const int count = bead_count(tag, 4, 1), stride = 4 / count;
                for (int bead = 0; bead < count; ++bead)
                    for (int d = 0; d < 3; ++d) {
                        double energies[2];
                        for (int side = 0; side < 2; ++side) {
                            if (lmp->universe->iworld / stride == bead)
                                for (int i = 0; i < lmp->atom->nlocal; ++i)
                                    if (lmp->atom->tag[i] == tag)
                                        lmp->atom->x[i][d] += (side ? -2 : 1) * 1.0e-6;
                            commands(lmp, "run 0");
                            energies[side] = fix->compute_vector(0);
                        }
                        if (lmp->universe->iworld / stride == bead)
                            for (int i = 0; i < lmp->atom->nlocal; ++i)
                                if (lmp->atom->tag[i] == tag) lmp->atom->x[i][d] += 1.0e-6;
                        commands(lmp, "run 0");
                        EXPECT_NEAR(initial[9 * (6 * bead * stride + tag - 1) + 6 + d] / count,
                                    -(energies[0] - energies[1]) / 2.0e-6, 2.0e-6);
                    }
            }
        }
        lammps_close(lmp);
    }
}

TEST(PIMDMixedMolecular, classical_limit_matches_native_nve)
{
    std::vector<double> reference, candidate;
    for (bool native : {true, false}) {
        auto *lmp = open_system(1);
        if (prepare(lmp, 1, 0, false, 2, native) && commands(lmp, "run 100")) {
            if (native)
                reference = state(lmp);
            else
                candidate = state(lmp);
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 54);
    ASSERT_EQ(candidate.size(), reference.size());
    for (std::size_t k = 0; k < reference.size(); ++k)
        EXPECT_NEAR(reference[k], candidate[k], 1.0e-10);
}

TEST(PIMDMixedMolecular, nve_energy_and_total_momentum)
{
    double errors[2] = {};
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(4);
        if (prepare(lmp, 4, 1, false, 2, false, true, true) &&
            commands(lmp, lane ? "timestep 0.1\nrun 0" : "timestep 0.2\nrun 0")) {
            auto *fix            = lmp->modify->get_fix_by_id("pi");
            const double initial = fix->compute_scalar();
            const auto p0        = momentum(lmp, 4, 1);
            for (int block = 0; block < 10; ++block) {
                if (!commands(lmp, lane ? "run 100 pre no post no" : "run 50 pre no post no"))
                    break;
                errors[lane] = std::max(errors[lane], std::abs(fix->compute_scalar() - initial));
            }
            const auto p1 = momentum(lmp, 4, 1);
            for (int d = 0; d < 3; ++d)
                EXPECT_NEAR(p0[d], p1[d], 2.0e-9);
            check_copies_and_geometry(lmp, 4, 1);
        }
        lammps_close(lmp);
    }
    EXPECT_GT(errors[0], 1.0e-8);
    EXPECT_LT(errors[1], 0.30 * errors[0]);
    EXPECT_LT(errors[0], 0.01);
}

TEST(PIMDMixedMolecular, two_and_four_water_pile_count_limits)
{
    for (int molecules : {2, 4})
        for (int mode : {0, 1, 2}) {
            auto *lmp = open_system(4);
            if (prepare(lmp, 4, mode, true, molecules) && commands(lmp, "run 1000")) {
                check_copies_and_geometry(lmp, 4, mode);
                auto *fix = lmp->modify->get_fix_by_id("pi");
                for (int k = 0; k < 6; ++k)
                    EXPECT_TRUE(std::isfinite(fix->compute_vector(k)));
            }
            lammps_close(lmp);
        }
}

TEST(PIMDMixedMolecular, deterministic_bonded_restart)
{
    restart_roundtrip(false);
}

TEST(PIMDMixedMolecular, pile_bonded_restart)
{
    restart_roundtrip(true);
}
