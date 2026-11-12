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
#include "update.h"

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

bool commands(LAMMPS *lmp, const std::string &text)
{
    lammps_commands_string(lmp, text.c_str());
    int invalid        = lammps_has_error(lmp);
    char message[1024] = {};
    if (invalid) lammps_get_last_error_message(lmp, message, sizeof(message));
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(invalid, 0) << message;
    return invalid == 0;
}

bool prepare(LAMMPS *lmp, int partitions, bool mixed = true, double offset = 0.0,
             bool tilted = false, double yshift = 0.0)
{
    const int b        = lmp->universe->iworld;
    const double shift = 0.02 * (b - 0.5 * (partitions - 1));
    std::string input  = "units lj\natom_style atomic\natom_modify map array sort 0 0\n";
    if (lmp->comm->nprocs == 2) input += "processors 2 1 1\n";
    input += "region box block 0 12 0 12 0 12\ncreate_box 1 box\n";
    input += "create_atoms 1 single " + std::to_string(5.0 + shift + offset) + " 5 5\n";
    input += "create_atoms 1 single " + std::to_string(6.2 + offset) + " 5 5\n";
    if (tilted) input += "change_box all triclinic\nchange_box all xy final 1.2\n";
    if (yshift != 0.0)
        input += "displace_atoms all move 0 " + std::to_string(yshift) + " 0 units box\n";
    input += "mass 1 2\npair_style lj/cut 2.5\npair_coeff * * 0.1 1.0\n";
    input += "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\ngroup quantum id 1\n";
    input += "velocity all set 0.01 0.02 -0.01\ntimestep 0.001\n";
    input += mixed ? "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads quantum " +
                         std::to_string(partitions) + "\n"
                   : "fix pi all nve\n";
    input += "thermo 1000\nrun 0\n";
    return commands(lmp, input);
}

std::vector<double> state(LAMMPS *lmp, bool unwrapped = false)
{
    std::vector<double> values(18 * lmp->universe->nworlds, 0.0);
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int base = 18 * lmp->universe->iworld + 9 * (lmp->atom->tag[i] - 1);
        double position[3];
        lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], position);
        for (int d = 0; d < 3; ++d) {
            values[base + d]     = unwrapped ? position[d] : lmp->atom->x[i][d];
            values[base + 3 + d] = lmp->atom->v[i][d];
            values[base + 6 + d] = lmp->atom->f[i][d];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return values;
}

double lj_force(double delta)
{
    const double inverse_r2 = 1.0 / (delta * delta);
    const double sr6        = inverse_r2 * inverse_r2 * inverse_r2;
    return 48 * 0.1 * inverse_r2 * sr6 * (sr6 - 0.5) * delta;
}

void check_force(double offset)
{
    auto *lmp = open_system(2);
    if (prepare(lmp, 2, true, offset)) {
        const auto result = state(lmp);
        const double f0 = lj_force(-1.21), f1 = lj_force(-1.19);
        EXPECT_NEAR(result[6], f0, 1.0e-12);
        EXPECT_NEAR(result[18 + 6], f1, 1.0e-12);
        EXPECT_NEAR(result[9 + 6], -0.5 * (f0 + f1), 1.0e-12);
        EXPECT_NEAR(result[27 + 6], -0.5 * (f0 + f1), 1.0e-12);
        auto *fix = lmp->modify->get_fix_by_id("pi");
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const bool quantum = lmp->atom->tag[i] == 1;
            EXPECT_DOUBLE_EQ(fix->array_atom[i][0], quantum ? 2.0 : 1.0);
            EXPECT_DOUBLE_EQ(fix->array_atom[i][2],
                             quantum || lmp->universe->iworld == 0 ? 1.0 : 0.0);
        }
    }
    lammps_close(lmp);
}
} // namespace

TEST(PIMDMixed, physical_force_pullback_multirank)
{
    check_force(0.0);
}

TEST(PIMDMixed, physical_force_pullback_empty_rank)
{
    check_force(-2.0);
}

TEST(PIMDMixed, classical_limit_matches_nve)
{
    std::vector<double> reference, candidate;
    for (bool mixed : {false, true}) {
        auto *lmp = open_system(1);
        if (prepare(lmp, 1, mixed) && commands(lmp, "run 50")) {
            if (mixed)
                candidate = state(lmp);
            else
                reference = state(lmp);
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 18);
    ASSERT_EQ(candidate.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], candidate[i], 1.0e-12);
}

TEST(PIMDMixed, classical_copies_remain_one_state)
{
    auto *lmp = open_system(4);
    if (prepare(lmp, 4) && commands(lmp, "run 100")) {
        const auto result = state(lmp);
        for (int b = 1; b < 4; ++b)
            for (int d = 0; d < 9; ++d)
                EXPECT_DOUBLE_EQ(result[9 + d], result[b * 18 + 9 + d]);
        EXPECT_GT(std::abs(result[0] - result[18 * 3]), 1.0e-5);
        const auto *fix = lmp->modify->get_fix_by_id("pi");
        EXPECT_EQ(fix->size_vector, 7);
    }
    lammps_close(lmp);
}

TEST(PIMDMixed, nve_energy_error_decreases_quadratically)
{
    double errors[2] = {};
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(4);
        if (prepare(lmp, 4) &&
            commands(lmp, lane == 0 ? "timestep 0.002\nrun 0" : "timestep 0.001\nrun 0")) {
            auto *fix            = lmp->modify->get_fix_by_id("pi");
            const double initial = fix->compute_scalar();
            for (int block = 0; block < 10; ++block) {
                if (!commands(lmp, lane == 0 ? "run 20 pre no post no" : "run 40 pre no post no"))
                    break;
                errors[lane] = std::max(errors[lane], std::abs(fix->compute_scalar() - initial));
            }
        }
        lammps_close(lmp);
    }
    EXPECT_GT(errors[0], 1.0e-10);
    EXPECT_LT(errors[1], 0.30 * errors[0]);
    EXPECT_LT(errors[0], 1.0e-4);
}

TEST(PIMDMixed, deterministic_restart_preserves_mixed_assignment)
{
    std::vector<double> reference, candidate;
    auto *continuous = open_system(2);
    if (prepare(continuous, 2) && commands(continuous, "run 40")) reference = state(continuous);
    lammps_close(continuous);
    auto *first            = open_system(2);
    const std::string file = "pimd_mixed.restart." + std::to_string(first->universe->iworld);
    if (prepare(first, 2) && commands(first, "run 20")) commands(first, "write_restart " + file);
    lammps_close(first);
    auto *second = open_system(2);
    if (commands(second,
                 "atom_modify map array sort 0 0\nread_restart " + file +
                     "\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
                     "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads quantum 2\nrun 20"))
        candidate = state(second);
    if (second->comm->me == 0) std::remove(file.c_str());
    lammps_close(second);
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(candidate.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], candidate[i], 1.0e-12);
}

TEST(PIMDMixed, inconsistent_classical_copies_fail)
{
    auto *lmp = open_system(2);
    if (prepare(lmp, 2)) {
        const double shift = lmp->universe->iworld == 0 ? 0.0 : 0.01;
        commands(lmp, "group classical id 2\ndisplace_atoms classical move " +
                          std::to_string(shift) + " 0 0 units box");
        lammps_command(lmp, "run 0");
        int invalid = lammps_has_error(lmp);
        int minimum = invalid;
        MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        EXPECT_EQ(minimum, 1);
        char message[1024] = {};
        lammps_get_last_error_message(lmp, message, sizeof(message));
        EXPECT_NE(std::string(message).find("matching positions and velocities"),
                  std::string::npos);
    }
    lammps_close(lmp);
}

namespace {
void expect_error(LAMMPS *lmp, const std::string &input, const std::string &fragment)
{
    lammps_commands_string(lmp, input.c_str());
    int invalid = lammps_has_error(lmp);
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(invalid, 1);
    char message[1024] = {};
    lammps_get_last_error_message(lmp, message, sizeof(message));
    EXPECT_NE(std::string(message).find(fragment), std::string::npos) << message;
}

void free_ring(const std::string &units, double dt)
{
    auto *lmp             = open_system(4);
    const int b           = lmp->universe->iworld;
    const double shape[4] = {1.0, 0.0, -1.0, 0.0};
    std::string input =
        "units " + units +
        "\natom_style atomic\natom_modify map array sort 0 0\n"
        "region box block 0 12 0 12 0 12\ncreate_box 1 box\ncreate_atoms 1 single " +
        std::to_string(5.0 + 0.1 * shape[b]) +
        " 5 5\nmass 1 2\npair_style zero 2\n"
        "pair_coeff * *\nvelocity all set 0 0 0\ntimestep " +
        std::to_string(dt) + "\nfix pi all pimd/mixed temp 300 beads all 4\nrun 0\n";
    if (commands(lmp, input)) {
        auto *fix            = lmp->modify->get_fix_by_id("pi");
        const double hbar    = lmp->force->hplanck / (2.0 * std::acos(-1.0));
        const double omega   = 4.0 * lmp->force->boltz * 300 / hbar * std::sqrt(2.0);
        const double initial = 0.5 * lmp->force->mvv2e * (2.0 / 4.0) * omega * omega * 0.02;
        EXPECT_NEAR(fix->compute_vector(1), initial, 1.0e-11 * std::max(1.0, initial));
        if (commands(lmp, "run 25")) {
            const double phase = omega * 25 * dt;
            EXPECT_NEAR(lmp->atom->x[0][0], 5.0 + 0.1 * shape[b] * std::cos(phase), 1.0e-12);
            EXPECT_NEAR(lmp->atom->v[0][0], -0.1 * shape[b] * omega * std::sin(phase),
                        1.0e-11 * std::max(1.0, omega));
            EXPECT_NEAR(fix->compute_scalar(), initial, 1.0e-10 * std::max(1.0, initial));
            EXPECT_NEAR(fix->compute_vector(1), initial * std::pow(std::cos(phase), 2),
                        1.0e-10 * std::max(1.0, initial));
            EXPECT_NEAR(fix->compute_vector(2), initial * std::pow(std::sin(phase), 2),
                        1.0e-10 * std::max(1.0, initial));
        }
    }
    lammps_close(lmp);
}

void periodic_translation(bool tilted)
{
    std::vector<double> reference, translated;
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(2);
        if (prepare(lmp, 2, true, 0.0, tilted, lane ? 6.99 : 0.0) &&
            commands(lmp, "velocity all set 0.01 1 -0.01\nrun 50")) {
            auto result = state(lmp, true);
            if (lane) {
                int crossed = 0;
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    if (((lmp->atom->image[i] >> IMGBITS) & IMGMASK) - IMGMAX != 0) crossed = 1;
                MPI_Allreduce(MPI_IN_PLACE, &crossed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                EXPECT_EQ(crossed, 1);
                for (int b = 0; b < 2; ++b)
                    for (int i = 0; i < 2; ++i)
                        result[18 * b + 9 * i + 1] -= 6.99;
                translated = result;
            } else
                reference = result;
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(translated.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], translated[i], 2.0e-11);
}
} // namespace

TEST(PIMDMixed, free_ring_real_units)
{
    free_ring("real", 0.1);
}
TEST(PIMDMixed, free_ring_metal_units)
{
    free_ring("metal", 0.0001);
}
TEST(PIMDMixed, periodic_crossing_orthogonal)
{
    periodic_translation(false);
}
TEST(PIMDMixed, periodic_crossing_triclinic)
{
    periodic_translation(true);
}

TEST(PIMDMixed, atom_storage_growth_preserves_tag_state)
{
    std::vector<double> reference, grown;
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(2);
        if (prepare(lmp, 2) && commands(lmp, "run 10")) {
            if (lane) lmp->atom->avec->grow(lmp->atom->nmax + 2000);
            if (commands(lmp, "run 30")) {
                if (lane)
                    grown = state(lmp);
                else
                    reference = state(lmp);
            }
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(grown.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], grown[i], 1.0e-12);
}

TEST(PIMDMixed, sorting_and_processor_migration_preserve_tag_state)
{
    std::vector<double> reference, sorted;
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(2);
        if (prepare(lmp, 2) && commands(lmp, "velocity all set 4 0.02 -0.01")) {
            int initial_owner = -1;
            for (int i = 0; i < lmp->atom->nlocal; ++i)
                if (lmp->atom->tag[i] == 1) initial_owner = lmp->comm->me;
            MPI_Allreduce(MPI_IN_PLACE, &initial_owner, 1, MPI_INT, MPI_MAX, lmp->world);
            if (lane) commands(lmp, "atom_modify sort 1 1.0");
            if (commands(lmp, "run 300")) {
                int final_owner = -1;
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    if (lmp->atom->tag[i] == 1) final_owner = lmp->comm->me;
                MPI_Allreduce(MPI_IN_PLACE, &final_owner, 1, MPI_INT, MPI_MAX, lmp->world);
                EXPECT_NE(initial_owner, final_owner);
                if (lane)
                    sorted = state(lmp);
                else
                    reference = state(lmp);
            }
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(sorted.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], sorted[i], 2.0e-12);
}

TEST(PIMDMixed, mismatched_timestep_fails_collectively)
{
    auto *lmp = open_system(2);
    if (prepare(lmp, 2)) {
        commands(lmp, lmp->universe->iworld == 0 ? "timestep 0.002" : "timestep 0.001");
        expect_error(lmp, "run 0", "matching positive timesteps");
    }
    lammps_close(lmp);
}

TEST(PIMDMixed, unsupported_fix_in_one_partition_fails_collectively)
{
    auto *lmp = open_system(2);
    if (prepare(lmp, 2)) {
        commands(lmp, lmp->universe->iworld == 0 ? "fix other all nve" : "variable unused equal 0");
        expect_error(lmp, "run 0", "companion fixes");
    }
    lammps_close(lmp);
}

TEST(PIMDMixed, changed_assignment_and_cell_fail)
{
    for (const bool cell : {false, true}) {
        auto *lmp = open_system(2);
        if (prepare(lmp, 2)) {
            commands(lmp, cell ? "change_box all x scale 1.01" : "group quantum id 2");
            expect_error(lmp, "run 0",
                         cell ? "unchanged simulation cell" : "assignment must match");
        }
        lammps_close(lmp);
    }
}

TEST(PIMDMixed, malformed_restart_payload_fails)
{
    auto *lmp = open_system(2);
    if (prepare(lmp, 2)) {
        auto *fix       = lmp->modify->get_fix_by_id("pi");
        double data[12] = {81753.0, 2.0, 2.0, 2.0, 1.0, 6.283185307179586,
                           0.0,     0.0, 0.0, 2.0, 2.0, 1.0};
        EXPECT_THROW(fix->restart(reinterpret_cast<char *>(data), 7 * sizeof(double)),
                     std::exception);
        data[11] = 1.5;
        EXPECT_THROW(fix->restart(reinterpret_cast<char *>(data), sizeof(data)), std::exception);
    }
    lammps_close(lmp);
}

namespace {
bool enable_pile(LAMMPS *lmp, int partitions)
{
    return commands(lmp,
                    "unfix pi\nfix pi all pimd/mixed temp 1 sp 6.283185307179586 beads quantum " +
                        std::to_string(partitions) + " thermostat pile 1 19771\nrun 0");
}
} // namespace

TEST(PIMDMixed, pile_classical_copies_have_one_noise_process)
{
    auto *lmp = open_system(4);
    if (prepare(lmp, 4) && enable_pile(lmp, 4) && commands(lmp, "run 100")) {
        const auto result = state(lmp);
        for (int b = 1; b < 4; ++b)
            for (int d = 0; d < 9; ++d)
                EXPECT_DOUBLE_EQ(result[9 + d], result[b * 18 + 9 + d]);
        EXPECT_GT(std::abs(result[12] - 0.01), 1.0e-5);
    }
    lammps_close(lmp);
}

TEST(PIMDMixed, pile_run_zero_does_not_consume_or_reset_random_state)
{
    std::vector<double> reference, repeated;
    for (int lane = 0; lane < 2; ++lane) {
        auto *lmp = open_system(2);
        if (prepare(lmp, 2) && enable_pile(lmp, 2) && commands(lmp, "run 21")) {
            if (lane) commands(lmp, "run 0\nrun 0");
            if (commands(lmp, "run 19")) {
                if (lane)
                    repeated = state(lmp);
                else
                    reference = state(lmp);
            }
        }
        lammps_close(lmp);
    }
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(repeated.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_DOUBLE_EQ(reference[i], repeated[i]);
}

TEST(PIMDMixed, pile_restart_restores_cached_gaussian)
{
    std::vector<double> reference, candidate;
    auto *continuous = open_system(2);
    if (prepare(continuous, 2) && enable_pile(continuous, 2) && commands(continuous, "run 40"))
        reference = state(continuous);
    lammps_close(continuous);
    auto *first            = open_system(2);
    const std::string file = "pimd_mixed_pile.restart." + std::to_string(first->universe->iworld);
    if (prepare(first, 2) && enable_pile(first, 2) && commands(first, "run 21"))
        commands(first, "write_restart " + file);
    lammps_close(first);
    auto *second = open_system(2);
    if (commands(second, "atom_modify map array sort 0 0\nprocessors 2 1 1\nread_restart " + file +
                             "\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
                             "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads quantum 2 "
                             "thermostat pile 1 19771\nrun 19"))
        candidate = state(second);
    if (second->comm->me == 0) std::remove(file.c_str());
    lammps_close(second);
    ASSERT_EQ(reference.size(), 36);
    ASSERT_EQ(candidate.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(reference[i], candidate[i], 1.0e-12);
}
