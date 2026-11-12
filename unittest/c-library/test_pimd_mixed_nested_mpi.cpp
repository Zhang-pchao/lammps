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
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "fix.h"
#include "force.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "universe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
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

std::string fix_command(int partitions, bool pile)
{
    return "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads fine " +
           std::to_string(partitions) + " beads coarse 2" +
           (pile ? " thermostat pile 1 19771" : "") + "\n";
}

bool prepare(LAMMPS *lmp, int partitions, bool pile = false, double offset = 0.0)
{
    const int b         = lmp->universe->iworld;
    const double fine   = 0.02 * (b - 0.5 * (partitions - 1));
    const double coarse = b < partitions / 2 ? -0.025 : 0.025;
    std::string input   = "units lj\natom_style atomic\natom_modify map array sort 0 0\n";
    if (lmp->comm->nprocs == 2) input += "processors 2 1 1\n";
    input += "region box block 0 12 0 12 0 12\ncreate_box 3 box\n";
    input += "create_atoms 1 single " + std::to_string(4.9 + fine + offset) + " 5 5\n";
    input += "create_atoms 2 single " + std::to_string(6.1 + coarse + offset) + " 5.1 5.2\n";
    input += "create_atoms 3 single " + std::to_string(7.3 + offset) + " 4.9 5.1\n";
    input += "mass 1 1\nmass 2 2\nmass 3 3\npair_style lj/cut 2.5\npair_coeff * * 0.1 1\n";
    input += "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n";
    input += "group fine id 1\ngroup coarse id 2\nvelocity all set 0.01 0.02 -0.01\n";
    input += "timestep 0.001\nthermo 100000\n" + fix_command(partitions, pile) + "run 0\n";
    return commands(lmp, input);
}

std::vector<double> state(LAMMPS *lmp)
{
    std::vector<double> values(27 * lmp->universe->nworlds, 0.0);
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int base = 27 * lmp->universe->iworld + 9 * (lmp->atom->tag[i] - 1);
        double x[3];
        lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], x);
        for (int d = 0; d < 3; ++d) {
            values[base + d]     = x[d];
            values[base + 3 + d] = lmp->atom->v[i][d];
            values[base + 6 + d] = lmp->atom->f[i][d];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return values;
}

// Independent full-configuration LJ energy; no production map or force helper.
double potential(const std::vector<double> &values, int partitions)
{
    double total = 0;
    for (int b = 0; b < partitions; ++b)
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) {
                double r2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double delta = values[27 * b + 9 * i + d] - values[27 * b + 9 * j + d];
                    r2 += delta * delta;
                }
                if (r2 < 6.25) {
                    const double sr6 = 1.0 / (r2 * r2 * r2);
                    total += 0.4 * sr6 * (sr6 - 1.0) / partitions;
                }
            }
    return total;
}

void check_copies(LAMMPS *lmp, int partitions)
{
    const auto values  = state(lmp);
    const int counts[] = {partitions, 2, 1};
    for (int i = 0; i < 3; ++i)
        for (int b = 0; b < partitions; ++b) {
            const int owner = (b / (partitions / counts[i])) * (partitions / counts[i]);
            for (int d = 0; d < 9; ++d)
                EXPECT_DOUBLE_EQ(values[27 * b + 9 * i + d], values[27 * owner + 9 * i + d]);
        }
    auto *fix   = lmp->modify->get_fix_by_id("pi");
    const int b = lmp->universe->iworld;
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int p = counts[lmp->atom->tag[i] - 1], stride = partitions / p;
        EXPECT_DOUBLE_EQ(fix->array_atom[i][0], p);
        EXPECT_DOUBLE_EQ(fix->array_atom[i][1], b / stride + 1);
        EXPECT_DOUBLE_EQ(fix->array_atom[i][2], b % stride == 0 ? 1 : 0);
    }
}

void expect_error(LAMMPS *lmp, const std::string &input, const std::string &fragment)
{
    lammps_commands_string(lmp, input.c_str());
    int bad = lammps_has_error(lmp);
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(bad, 1);
    char message[1024] = {};
    lammps_get_last_error_message(lmp, message, sizeof(message));
    EXPECT_NE(std::string(message).find(fragment), std::string::npos) << message;
}

void restart_roundtrip(bool pile)
{
    for (int partitions : {4, 8}) {
        std::vector<double> reference, resumed;
        auto *continuous = open_system(partitions);
        if (prepare(continuous, partitions, pile) && commands(continuous, "run 40"))
            reference = state(continuous);
        lammps_close(continuous);
        auto *first            = open_system(partitions);
        const std::string file = "mixed_nested.restart." + std::to_string(partitions) + "." +
                                 std::to_string(first->universe->iworld);
        if (prepare(first, partitions, pile) && commands(first, "run 21"))
            commands(first, "write_restart " + file);
        lammps_close(first);
        auto *second      = open_system(partitions);
        std::string input = "atom_modify map array sort 0 0\n";
        if (second->comm->nprocs == 2) input += "processors 2 1 1\n";
        input +=
            "read_restart " + file + "\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n";
        if (commands(second, input + fix_command(partitions, pile) + "run 19")) {
            resumed = state(second);
            check_copies(second, partitions);
        }
        if (second->comm->me == 0) std::remove(file.c_str());
        lammps_close(second);
        ASSERT_EQ(reference.size(), 27 * partitions);
        ASSERT_EQ(resumed.size(), reference.size());
        for (std::size_t k = 0; k < reference.size(); ++k)
            EXPECT_NEAR(reference[k], resumed[k], 2.0e-12);
    }
}
} // namespace

TEST(PIMDMixedNested, force_finite_differences_and_metadata)
{
    for (int partitions : {4, 8})
        for (double offset : {0.0, -3.0}) {
            auto *lmp = open_system(partitions);
            if (prepare(lmp, partitions, false, offset)) {
                const auto values = state(lmp);
                EXPECT_NEAR(lmp->modify->get_fix_by_id("pi")->compute_vector(0),
                            potential(values, partitions), 1.0e-12);
                const int counts[] = {partitions, 2, 1};
                for (int atom = 0; atom < 3; ++atom)
                    for (int bead = 0; bead < counts[atom]; ++bead)
                        for (int d = 0; d < 3; ++d) {
                            auto plus = values, minus = values;
                            const int begin = bead * partitions / counts[atom];
                            const int end   = (bead + 1) * partitions / counts[atom];
                            for (int b = begin; b < end; ++b) {
                                plus[27 * b + 9 * atom + d] += 1.0e-6;
                                minus[27 * b + 9 * atom + d] -= 1.0e-6;
                            }
                            const double derivative =
                                -(potential(plus, partitions) - potential(minus, partitions)) /
                                2.0e-6;
                            for (int b = begin; b < end; ++b)
                                EXPECT_NEAR(values[27 * b + 9 * atom + 6 + d] / counts[atom],
                                            derivative, 1.0e-9);
                        }
                check_copies(lmp, partitions);
                if (partitions == 4 && offset < 0) {
                    int empty = lmp->atom->nlocal == 0;
                    MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                    EXPECT_EQ(empty, 4);
                }
            }
            lammps_close(lmp);
        }
}

TEST(PIMDMixedNested, nve_energy_error_decreases_quadratically)
{
    double errors[2] = {};
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(8);
        if (prepare(lmp, 8) &&
            commands(lmp, lane ? "timestep 0.001\nrun 0" : "timestep 0.002\nrun 0")) {
            auto *fix            = lmp->modify->get_fix_by_id("pi");
            const double initial = fix->compute_scalar();
            for (int block = 0; block < 10; ++block) {
                if (!commands(lmp, lane ? "run 40 pre no post no" : "run 20 pre no post no")) break;
                errors[lane] = std::max(errors[lane], std::abs(fix->compute_scalar() - initial));
            }
            check_copies(lmp, 8);
        }
        lammps_close(lmp);
    }
    EXPECT_GT(errors[0], 1.0e-10);
    EXPECT_LT(errors[1], 0.30 * errors[0]);
    EXPECT_LT(errors[0], 1.0e-4);
}

TEST(PIMDMixedNested, pile_intermediate_copies_have_one_state)
{
    auto *lmp = open_system(8);
    if (prepare(lmp, 8, true) && commands(lmp, "run 100")) {
        check_copies(lmp, 8);
        const auto values = state(lmp);
        EXPECT_GT(std::abs(values[9] - values[4 * 27 + 9]), 1.0e-5);
        EXPECT_GT(std::abs(values[18 + 3] - 0.01), 1.0e-5);
    }
    lammps_close(lmp);
}

TEST(PIMDMixedNested, deterministic_restart)
{
    restart_roundtrip(false);
}

TEST(PIMDMixedNested, pile_restart)
{
    restart_roundtrip(true);
}

TEST(PIMDMixedNested, invalid_intermediate_copies_fail_collectively)
{
    auto *lmp = open_system(8);
    if (prepare(lmp, 8)) {
        const double delta = lmp->universe->iworld == 1 ? 0.01 : 0;
        commands(lmp, "displace_atoms coarse move " + std::to_string(delta) + " 0 0 units box");
        expect_error(lmp, "run 0", "matching positions and velocities");
    }
    lammps_close(lmp);
}

TEST(PIMDMixedNested, invalid_counts_fail_before_conversion)
{
    auto *lmp = open_system(8);
    if (prepare(lmp, 8)) {
        auto *fix       = lmp->modify->get_fix_by_id("pi");
        double data[13] = {81753.0, 2.0, 8.0, 3.0, 1.0, 6.283185307179586, 0.0, 0.0,
                           0.0,     1.0, 8.0, 2.0, 1.0};
        for (double bad :
             {-1.0, 0.0, 1.5, 3.0, 16.0, 1.0e100, std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::quiet_NaN()}) {
            data[11] = bad;
            EXPECT_THROW(fix->restart(reinterpret_cast<char *>(data), sizeof(data)),
                         std::exception);
        }
        expect_error(lmp, "unfix pi\nfix pi all pimd/mixed temp 1 beads fine 8 beads coarse 3",
                     "powers of two");
    }
    lammps_close(lmp);
}

TEST(PIMDMixedNested, sorting_migration_and_growth_preserve_deterministic_state)
{
    std::vector<double> reference, candidate;
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(4);
        if (prepare(lmp, 4) && commands(lmp, "velocity all set 4 0.02 -0.01")) {
            int first = -1;
            for (int i = 0; i < lmp->atom->nlocal; ++i)
                if (lmp->atom->tag[i] == 1) first = lmp->comm->me;
            MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MAX, lmp->world);
            if (lane) {
                lmp->atom->avec->grow(lmp->atom->nmax + 2000);
                commands(lmp, "atom_modify sort 1 1.0");
            }
            if (commands(lmp, "run 350")) {
                int last = -1;
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    if (lmp->atom->tag[i] == 1) last = lmp->comm->me;
                MPI_Allreduce(MPI_IN_PLACE, &last, 1, MPI_INT, MPI_MAX, lmp->world);
                EXPECT_NE(first, last);
                check_copies(lmp, 4);
                if (lane)
                    candidate = state(lmp);
                else
                    reference = state(lmp);
            }
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 108);
    ASSERT_EQ(candidate.size(), reference.size());
    for (std::size_t k = 0; k < reference.size(); ++k)
        EXPECT_NEAR(reference[k], candidate[k], 2.0e-11);
}
