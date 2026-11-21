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

#define LAMMPS_LIB_MPI 1
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "fix.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "neighbor.h"
#include "REPLICA/pimd_pes_view.h"
#include "universe.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;

namespace {
std::string number(double x)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << x;
    return stream.str();
}
LAMMPS *open_system(int p)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string layout = std::to_string(p) + "x" + std::to_string(size / p);
    const char *args[]       = {"LAMMPS_test", "-screen",      "none", "-log", "none",
                                "-partition",  layout.c_str(), "-in",  "none", "-nocite"};
    return static_cast<LAMMPS *>(
        lammps_open(10, const_cast<char **>(args), MPI_COMM_WORLD, nullptr));
}
bool commands(LAMMPS *lmp, const std::string &input)
{
    lammps_commands_string(lmp, input.c_str());
    int invalid        = lammps_has_error(lmp);
    char message[1024] = {};
    if (invalid) lammps_get_last_error_message(lmp, message, sizeof(message));
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(invalid, 0) << message;
    return invalid == 0;
}
std::string options(const std::string &region, bool pile)
{
    std::string text = "fix pi all pimd/adaptive temp 1 sp 6.283185307179586 center ";
    if ((region == "mobile") || (region == "mobile-slab") || (region == "mobile-all"))
        text += "group site core 0.2 outer 2.4";
    else if (region == "quantum")
        text += "5.6 5 5 core 3 outer 4";
    else if ((region == "classical") || (region.find("protected") == 0))
        text += "0 0 0 core 0.5 outer 1";
    else
        text += "3 5 5 core 0.5 outer 4";
    if ((region == "slab") || (region == "mobile-slab")) text += " geometry slab x";
    if (region.find("protected") == 0) text += " quantum site";
    text += " initialize auxiliary";
    if (pile) text += " thermostat pile 1 19771";
    return text + "\n";
}
bool prepare(LAMMPS *lmp, const std::string &region = "transition", bool pile = false,
             int perturb = -1, double displacement = 0.0,
             const std::string &integrator = "adaptive", double dt = 0.001)
{
    const int p = lmp->universe->nworlds, b = lmp->universe->iworld;
    const double shift = 0.04 * (b - 0.5 * (p - 1));
    std::string input  = "units lj\natom_style atomic\natom_modify map array sort 0 0\n";
    if (lmp->comm->nprocs == 2) input += "processors * 1 1\n";
    input += "region box block 0 12 0 12 0 12\ncreate_box 1 box\n";
    for (int i = 0; i < 2; ++i) {
        double x[3] = {5.0 + 1.2 * i + shift * (1.0 - 0.5 * i), 5.0 + shift * 0.3,
                       5.0 - shift * 0.2};
        for (int d = 0; d < 3; ++d)
            if (perturb == (b * 2 + i) * 3 + d) x[d] += displacement;
        input += "create_atoms 1 single " + number(x[0]) + " " + number(x[1]) + " " + number(x[2]) +
                 "\n";
    }
    input += "mass 1 2\npair_style lj/cut 2.5\npair_coeff * * 0.1 1\n"
             "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
             "velocity all set 0.01 0.02 -0.01\ntimestep " +
             number(dt) + "\n";
    if (region.find("mobile") == 0) {
        input += region == "mobile-all" ? "group site id 1 2\n" : "group site id 1\n";
        input += "velocity site set 0.03 -0.01 0.02\n";
    }
    if (region.find("protected") == 0)
        input += region == "protected-all" ? "group site id 1 2\n" : "group site id 1\n";
    if (integrator == "adaptive")
        input += options(region, pile);
    else if (integrator == "mixed")
        input += "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads all " +
                 std::to_string(p) + "\n";
    else
        input += "fix pi all nve\n";
    input += "thermo 100000\nrun 0\n";
    return commands(lmp, input);
}
std::vector<double> state(LAMMPS *lmp, bool auxiliary = true)
{
    const int columns = auxiliary ? 22 : 9;
    std::vector<double> values(columns * 2 * lmp->universe->nworlds, 0.0);
    auto *fix = lmp->modify->get_fix_by_id("pi");
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int base = columns * (2 * lmp->universe->iworld + lmp->atom->tag[i] - 1);
        double x[3];
        lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], x);
        for (int d = 0; d < 3; ++d) {
            values[base + d]     = x[d];
            values[base + 3 + d] = lmp->atom->v[i][d];
            values[base + 6 + d] = lmp->atom->f[i][d];
        }
        if (auxiliary)
            for (int d = 0; d < 13; ++d)
                values[base + 9 + d] = fix->array_atom[i][d];
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return values;
}
void equal(const std::vector<double> &a, const std::vector<double> &b, double tolerance = 1.0e-10)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], tolerance) << i;
}
void restart_case(bool pile, const std::string &region = "transition")
{
    auto *continuous = open_system(4);
    std::vector<double> reference, candidate;
    if (prepare(continuous, region, pile) && commands(continuous, "run 40"))
        reference = state(continuous);
    lammps_close(continuous);
    auto *first            = open_system(4);
    const std::string file = "pimd_adaptive." + std::to_string(pile) + ".restart." +
                             std::to_string(first->universe->iworld);
    if (prepare(first, region, pile) && commands(first, "run 21"))
        commands(first, "write_restart " + file);
    lammps_close(first);
    auto *second      = open_system(4);
    std::string input = "atom_modify map array sort 0 0\nprocessors * 1 1\nread_restart " + file +
                        "\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n" +
                        options(region, pile) + "run 19";
    if (commands(second, input)) candidate = state(second);
    if (second->comm->me == 0) std::remove(file.c_str());
    lammps_close(second);
    ASSERT_EQ(reference.size(), 176);
    equal(reference, candidate);
}
} // namespace

TEST(PIMDAdaptive, physical_and_auxiliary_force_finite_differences)
{
    auto *reference = open_system(4);
    std::vector<double> expected;
    if (prepare(reference)) expected = state(reference);
    lammps_close(reference);
    ASSERT_EQ(expected.size(), 176);
    for (int coordinate = 0; coordinate < 24; ++coordinate) {
        double energy[2] = {};
        for (int s = 0; s < 2; ++s) {
            auto *lmp = open_system(4);
            if (prepare(lmp, "transition", false, coordinate, s ? 1.0e-5 : -1.0e-5))
                energy[s] = lmp->modify->get_fix_by_id("pi")->compute_vector(0);
            lammps_close(lmp);
        }
        const int row = coordinate / 3, d = coordinate % 3;
        EXPECT_NEAR(expected[22 * row + 15 + d], -4 * (energy[1] - energy[0]) / 2.0e-5, 2.0e-8);
    }
}

TEST(PIMDAdaptive, quantum_endpoint_matches_fixed_uniform)
{
    std::vector<double> a, b;
    auto *lmp = open_system(4);
    if (prepare(lmp, "quantum") && commands(lmp, "run 50")) a = state(lmp, false);
    lammps_close(lmp);
    lmp = open_system(4);
    if (prepare(lmp, "quantum", false, -1, 0.0, "mixed") && commands(lmp, "run 50"))
        b = state(lmp, false);
    lammps_close(lmp);
    ASSERT_EQ(a.size(), 72);
    equal(a, b);
}

TEST(PIMDAdaptive, classical_endpoint_matches_md)
{
    std::vector<double> a, b;
    auto *lmp = open_system(4);
    if (prepare(lmp, "classical") && commands(lmp, "run 50")) a = state(lmp, false);
    lammps_close(lmp);
    lmp = open_system(1);
    if (prepare(lmp, "classical", false, -1, 0.0, "md") && commands(lmp, "run 50"))
        b = state(lmp, false);
    lammps_close(lmp);
    ASSERT_EQ(a.size(), 72);
    ASSERT_EQ(b.size(), 18);
    for (int p = 0; p < 4; ++p)
        for (int i = 0; i < 18; ++i)
            EXPECT_NEAR(a[p * 18 + i], b[i], 1.0e-10);
}

TEST(PIMDAdaptive, repeated_run_zero_and_array_growth)
{
    auto *lmp = open_system(4);
    if (prepare(lmp)) {
        const auto before = state(lmp);
        lmp->atom->avec->grow(lmp->atom->nmax + 100);
        if (commands(lmp, "run 0\nrun 0")) equal(before, state(lmp));
    }
    lammps_close(lmp);
}

TEST(PIMDAdaptive, conservative_transition_energy)
{
    double errors[2] = {};
    for (int level = 0; level < 2; ++level) {
        auto *lmp = open_system(4);
        if (prepare(lmp, "transition", false, -1, 0.0, "adaptive", level ? 0.001 : 0.002)) {
            auto *fix          = lmp->modify->get_fix_by_id("pi");
            const double first = fix->compute_scalar();
            if (commands(lmp, level ? "run 400" : "run 200"))
                errors[level] = std::abs(fix->compute_scalar() - first);
        }
        lammps_close(lmp);
    }
    EXPECT_GT(errors[0], 1.0e-12);
    EXPECT_LT(errors[0], 1.0e-4);
    EXPECT_LT(errors[1], 0.3 * errors[0]);
}
TEST(PIMDAdaptive, deterministic_restart)
{
    restart_case(false);
}
TEST(PIMDAdaptive, pile_restart_preserves_hidden_state_and_stream)
{
    restart_case(true);
}

TEST(PIMDAdaptive, free_path_crosses_regions_domains_and_periodic_boundary)
{
    auto *lmp             = open_system(4);
    const int b           = lmp->universe->iworld;
    const double shape[4] = {1, 0, -1, 0};
    std::string input =
        "units lj\natom_style atomic\natom_modify map array sort 1 0.4\n"
        "processors * 1 1\nregion box block 0 12 0 12 0 12\ncreate_box 1 box\n"
        "create_atoms 1 single " +
        number(3.0 + 0.1 * shape[b]) +
        " 5 5\n"
        "mass 1 2\npair_style zero 2\npair_coeff * *\nneighbor 0.3 bin\n"
        "neigh_modify every 1 delay 0 check no\nvelocity all set 2 0 0\ntimestep 0.005\n"
        "fix pi all pimd/adaptive temp 1 sp 6.283185307179586 center 6 5 5 core 0.4 outer 2 "
        "initialize auxiliary\nthermo 100000\nrun 0";
    if (commands(lmp, input)) {
        auto *fix           = lmp->modify->get_fix_by_id("pi");
        const double energy = fix->compute_scalar();
        double maximum      = 0.0;
        for (int block = 1; block <= 100; ++block) {
            if (!commands(lmp, "run 10 pre no post no")) break;
            double mean[4] = {};
            for (int i = 0; i < lmp->atom->nlocal; ++i) {
                double q[3];
                lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], q);
                mean[0] += q[0] / 4;
                mean[1] += fix->array_atom[i][0] / 4;
                mean[2] += lmp->atom->v[i][0] / 4;
                mean[3] += fix->array_atom[i][9] / 4;
            }
            MPI_Allreduce(MPI_IN_PLACE, mean, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            EXPECT_NEAR(mean[0], 3.0 + 0.1 * block, 2.0e-10);
            EXPECT_NEAR(mean[1], mean[0], 2.0e-10);
            EXPECT_NEAR(mean[2], 2.0, 2.0e-10);
            EXPECT_NEAR(fix->compute_scalar(), energy, 2.0e-10);
            maximum = std::max(maximum, mean[3]);
        }
        EXPECT_GT(maximum, 0.999);
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
} // namespace
TEST(PIMDAdaptive, unsupported_changes_fail_collectively)
{
    auto *lmp = open_system(4);
    if (prepare(lmp))
        expect_error(lmp, "displace_atoms all move 0.1 0 0 units box\nrun 0", "changed outside");
    lammps_close(lmp);
    lmp = open_system(4);
    if (prepare(lmp)) expect_error(lmp, "fix other all nve\nrun 0", "companion fixes");
    lammps_close(lmp);
    lmp = open_system(4);
    if (prepare(lmp))
        expect_error(lmp,
                     "unfix pi\nfix pi all pimd/adaptive temp 1 center 0 0 0 core 1 outer 6 "
                     "initialize auxiliary\nrun 0",
                     "periodic half-box");
    lammps_close(lmp);
}

TEST(PIMDAdaptive, mobile_and_slab_energy_velocity_derivatives)
{
    for (const std::string region : {"mobile", "mobile-all", "mobile-slab", "slab", "protected"}) {
        auto *reference = open_system(4);
        std::vector<double> expected;
        if (prepare(reference, region)) expected = state(reference);
        lammps_close(reference);
        ASSERT_EQ(expected.size(), 176);
        std::vector<double> tangent(24, 0.0);
        for (int coordinate = 0; coordinate < 24; ++coordinate) {
            double energy[2] = {};
            std::vector<double> states[2];
            for (int side = 0; side < 2; ++side) {
                auto *lmp = open_system(4);
                if (prepare(lmp, region, false, coordinate, side ? 1.0e-5 : -1.0e-5)) {
                    energy[side] = lmp->modify->get_fix_by_id("pi")->compute_vector(0);
                    states[side] = state(lmp);
                }
                lammps_close(lmp);
            }
            ASSERT_EQ(states[0].size(), 176);
            ASSERT_EQ(states[1].size(), 176);
            const int atom = coordinate / 3, d = coordinate % 3;
            EXPECT_NEAR(expected[22 * atom + 15 + d], -4 * (energy[1] - energy[0]) / 2.0e-5, 2.0e-8)
                << region << " coordinate " << coordinate;
            for (int j = 0; j < 24; ++j) {
                const int slot = 22 * (j / 3) + j % 3;
                tangent[j] +=
                    (states[1][slot] - states[0][slot]) / 2.0e-5 * expected[22 * atom + 12 + d];
            }
        }
        for (int j = 0; j < 24; ++j)
            EXPECT_NEAR(expected[22 * (j / 3) + 3 + j % 3], tangent[j], 1.0e-10) << region;
        if (region.find("mobile") == 0)
            for (int d = 0; d < 3; ++d) {
                double total = 0.0;
                for (int j = 0; j < 8; ++j)
                    total += expected[22 * j + 15 + d];
                EXPECT_NEAR(total, 0.0, 1.0e-12) << region;
            }
    }
}

TEST(PIMDAdaptive, mobile_slab_restart)
{
    restart_case(false, "mobile-slab");
    restart_case(true, "mobile-slab");
}

TEST(PIMDAdaptive, mobile_membership_changes_rejected)
{
    auto *lmp = open_system(4);
    if (prepare(lmp, "mobile"))
        expect_error(lmp, "group site id 2\nrun 0", "unchanged in all partitions");
    lammps_close(lmp);
}

TEST(PIMDAdaptive, compact_reference_crosses_periodic_boundary_with_unequal_masses)
{
    auto *lmp          = open_system(4);
    const double shift = 0.02 * (lmp->universe->iworld - 1.5);
    std::string input =
        "units lj\natom_style atomic\natom_modify map array sort 0 0\n"
        "processors * 1 1\nregion box block 0 12 0 12 0 12\ncreate_box 2 box\n"
        "create_atoms 1 single " +
        number(11.6 + shift) +
        " 5 5\n"
        "create_atoms 2 single " +
        number(0.6 + shift) +
        " 5 5\n"
        "mass 1 2\nmass 2 5\npair_style zero 2\npair_coeff * *\n"
        "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
        "group site id 1 2\nvelocity all set 2 0 0\ntimestep 0.005\n"
        "fix pi all pimd/adaptive temp 1 sp 6.283185307179586 center group site "
        "geometry slab x core 0.1 outer 2 initialize auxiliary\nthermo 100000\nrun 0";
    if (commands(lmp, input)) {
        const auto initial  = state(lmp);
        auto *fix           = lmp->modify->get_fix_by_id("pi");
        const double energy = fix->compute_scalar();
        for (int block = 1; block <= 40; ++block) {
            if (!commands(lmp, "run 10 pre no post no")) break;
            const auto current = state(lmp);
            for (int i = 0; i < 2; ++i) {
                double centroid = 0;
                for (int b = 0; b < 4; ++b) {
                    centroid += current[22 * (2 * b + i)] / 4;
                    EXPECT_NEAR(current[22 * (2 * b + i) + 18], initial[22 * (2 * b + i) + 18],
                                1.0e-11);
                }
                EXPECT_NEAR(centroid, (i == 0 ? 11.6 : 0.6) + 0.1 * block, 2.0e-10);
            }
            EXPECT_NEAR(fix->compute_scalar(), energy, 2.0e-10);
        }
    }
    lammps_close(lmp);
}

TEST(PIMDAdaptive, protected_fragment_restores_uniform_endpoint_and_restart)
{
    std::vector<double> a, b;
    auto *lmp = open_system(4);
    if (prepare(lmp, "protected-all") && commands(lmp, "run 50")) a = state(lmp, false);
    lammps_close(lmp);
    lmp = open_system(4);
    if (prepare(lmp, "quantum", false, -1, 0.0, "mixed") && commands(lmp, "run 50"))
        b = state(lmp, false);
    lammps_close(lmp);
    ASSERT_EQ(a.size(), 72);
    equal(a, b);
    restart_case(false, "protected");
    restart_case(true, "protected");
    lmp = open_system(4);
    if (prepare(lmp, "protected"))
        expect_error(lmp, "group site id 2\nrun 0", "unchanged in all partitions");
    lammps_close(lmp);
}

TEST(PIMDAdaptive, displacement_checked_neighbors)
{
    for (const std::string region :
         {"transition", "quantum", "classical", "mobile-slab", "protected"}) {
        std::vector<double> reference, candidate;
        auto *lmp = open_system(4);
        if (prepare(lmp, region, false, -1, 0.0, "adaptive", 0.005) &&
            commands(lmp, "neighbor 0.1 bin\nrun 200"))
            reference = state(lmp);
        lammps_close(lmp);

        lmp = open_system(4);
        if (prepare(lmp, region, false, -1, 0.0, "adaptive", 0.005) &&
            commands(lmp, "neighbor 0.1 bin\nneigh_modify every 1 delay 0 check yes\nrun 200")) {
            candidate = state(lmp);
            EXPECT_LT(lmp->neighbor->ncalls, 200) << region;
        }
        lammps_close(lmp);
        ASSERT_EQ(reference.size(), 176);
        equal(reference, candidate, 1.0e-10);
    }
}

TEST(PIMDAdaptive, potential_view_is_fresh_collective_and_tagged)
{
    for (const std::string region : {"transition", "classical", "quantum", "mobile", "protected"}) {
        auto *lmp = open_system(4);
        if (prepare(lmp, region)) {
            auto *fix                         = lmp->modify->get_fix_by_id("pi");
            std::uint64_t previous_generation = 0;
            for (int refresh = 0; refresh < 3; ++refresh) {
                const auto before = state(lmp);
                int dim           = -1;
                const auto *view =
                    static_cast<const PIMDPESView *>(fix->extract("pimd_pes_view_v1", dim));
                ASSERT_NE(view, nullptr);
                EXPECT_EQ(dim, 0);
                EXPECT_EQ(view->version, PIMDPESView::VERSION);
                EXPECT_EQ(view->beads, 4);
                EXPECT_EQ(view->bead, lmp->universe->iworld);
                EXPECT_EQ(view->nlocal, lmp->atom->nlocal);
                EXPECT_EQ(view->natoms, lmp->atom->natoms);
                EXPECT_GT(view->generation, previous_generation);
                previous_generation = view->generation;
                std::vector<double> reference(3 * lmp->atom->natoms, 0.0);
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    for (int d = 0; d < 3; ++d)
                        reference[3 * (lmp->atom->tag[i] - 1) + d] += fix->array_atom[i][d] / 4;
                MPI_Allreduce(MPI_IN_PLACE, reference.data(), reference.size(), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                for (int i = 0; i < view->nlocal; ++i) {
                    EXPECT_EQ(view->tags[i], lmp->atom->tag[i]);
                    EXPECT_DOUBLE_EQ(view->amplitude[i], fix->array_atom[i][9]);
                    for (int d = 0; d < 3; ++d)
                        EXPECT_NEAR(view->centroid[3 * i + d],
                                    reference[3 * (view->tags[i] - 1) + d], 1e-13);
                }
                equal(before, state(lmp));
                if (!commands(lmp, refresh == 0 ? "run 0" : "run 1")) break;
            }
        }
        lammps_close(lmp);
    }
}

TEST(PIMDAdaptive, potential_snapshot_checks_all_beads_and_local_order)
{
    const char *previous = std::getenv("LAMMPS_PIMD_PES_SNAPSHOT_EXPERIMENT");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    setenv("LAMMPS_PIMD_PES_SNAPSHOT_EXPERIMENT", "1", 1);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int p : {1, 2, 4}) {
        if ((p > size) || (size % p)) continue;
        auto *lmp = open_system(p);
        if (prepare(lmp)) {
            auto *fix = lmp->modify->get_fix_by_id("pi");
            int dim = 0;
            auto *counts = static_cast<double *>(fix->extract("pimd_pes_snapshot_counts", dim));
            ASSERT_NE(counts, nullptr);
            EXPECT_EQ(dim, 1);
            const bool single = lmp->comm->nprocs == 1;
            for (int phase = 0; phase < 5; ++phase) {
                // Reorder on alternating beads, without calling run/map_positions.
                if ((phase == 1) && single && (lmp->universe->iworld % 2)) {
                    lmp->atom->avec->grow(lmp->atom->nmax + 100);
                    lmp->atom->avec->copy(0, 2, 0);
                    lmp->atom->avec->copy(1, 0, 0);
                    lmp->atom->avec->copy(2, 1, 0);
                    lmp->atom->map_set();
                }
                // Only one bead mutates: every other bead must also fall back.
                if ((phase == 2) && (lmp->universe->iworld == 0))
                    for (int i = 0; i < lmp->atom->nlocal; ++i)
                        fix->array_atom[i][0] += 0.125;
                if ((phase == 3) && (lmp->universe->iworld == 0))
                    for (int i = 0; i < lmp->atom->nlocal; ++i)
                        fix->array_atom[i][0] -= 0.125;
                if (phase == 4) EXPECT_TRUE(commands(lmp, "run 0\nrun 2"));
                const auto before = state(lmp);
                const double hits = counts[0], misses = counts[1];
                const auto *view =
                    static_cast<const PIMDPESView *>(fix->extract("pimd_pes_view_v1", dim));
                const bool reuse = single && (phase != 2);
                EXPECT_DOUBLE_EQ(counts[0] - hits, reuse ? 1.0 : 0.0);
                EXPECT_DOUBLE_EQ(counts[1] - misses, reuse ? 0.0 : 1.0);
                std::vector<double> reference(3 * lmp->atom->natoms, 0.0);
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    for (int d = 0; d < 3; ++d)
                        reference[3 * (lmp->atom->tag[i] - 1) + d] = fix->array_atom[i][d] / p;
                MPI_Allreduce(MPI_IN_PLACE, reference.data(), reference.size(), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                for (int i = 0; i < view->nlocal; ++i) {
                    EXPECT_EQ(view->tags[i], lmp->atom->tag[i]);
                    EXPECT_DOUBLE_EQ(view->amplitude[i], fix->array_atom[i][9]);
                    for (int d = 0; d < 3; ++d)
                        EXPECT_NEAR(view->centroid[3 * i + d],
                                    reference[3 * (view->tags[i] - 1) + d], 1e-13);
                }
                equal(before, state(lmp), 0.0);
            }
        }
        lammps_close(lmp);
    }
    if (had_previous)
        setenv("LAMMPS_PIMD_PES_SNAPSHOT_EXPERIMENT", saved.c_str(), 1);
    else
        unsetenv("LAMMPS_PIMD_PES_SNAPSHOT_EXPERIMENT");
}

TEST(PIMDAdaptive, single_rank_collection_tracks_independent_atom_order)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto *reference = open_system(size);
    std::vector<double> expected;
    if (prepare(reference) && commands(reference, "run 20")) expected = state(reference);
    lammps_close(reference);

    auto *lmp = open_system(size);
    if (prepare(lmp)) {
        ASSERT_EQ(lmp->atom->nlocal, 2);
        const auto initial = state(lmp);
        lmp->atom->avec->grow(lmp->atom->nmax + 100);
        if (lmp->universe->iworld % 2) {
            lmp->atom->avec->copy(0, 2, 0);
            lmp->atom->avec->copy(1, 0, 0);
            lmp->atom->avec->copy(2, 1, 0);
            lmp->atom->map_set();
        }
        if (commands(lmp, "run 0")) equal(initial, state(lmp), 1.0e-12);
        if (commands(lmp, "run 20")) equal(expected, state(lmp), 1.0e-12);
    }
    lammps_close(lmp);
}

TEST(PIMDAdaptive, single_rank_nonconsecutive_tags_are_rejected)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto *lmp = open_system(size);
    if (prepare(lmp)) {
        lmp->atom->tag[1] = 3;
        lmp->atom->map_init();
        lmp->atom->map_set();
        expect_error(lmp, "run 0", "consecutive IDs");
    }
    lammps_close(lmp);
}


TEST(PIMDAdaptive, force_summary_matches_gather_with_reordered_atoms)
{
    const char *name = "LAMMPS_PIMD_FORCE_SUMMARY_EXPERIMENT";
    const char *previous = std::getenv(name);
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int p : {1, 2, 4, 16}) {
        if ((p > size) || (size % p)) continue;
        for (const std::string region : {"transition", "classical", "quantum", "mobile", "protected"}) {
            std::vector<double> reference, candidate;
            for (int enabled : {0, 1}) {
                setenv(name, enabled ? "1" : "0", 1);
                auto *lmp = open_system(p);
                if (prepare(lmp, region)) {
                    const bool single = lmp->comm->nprocs == 1;
                    if (single && (lmp->universe->iworld % 2)) {
                        lmp->atom->avec->grow(lmp->atom->nmax + 100);
                        lmp->atom->avec->copy(0, 2, 0);
                        lmp->atom->avec->copy(1, 0, 0);
                        lmp->atom->avec->copy(2, 1, 0);
                        lmp->atom->map_set();
                    }
                    if (commands(lmp, "run 0\nrun 20")) {
                        (enabled ? candidate : reference) = state(lmp);
                        int dim = 0;
                        auto *counts = static_cast<double *>(
                            lmp->modify->get_fix_by_id("pi")->extract("pimd_force_summary_counts", dim));
                        EXPECT_NE(counts, nullptr);
                        if (counts) {
                            EXPECT_GT(counts[enabled && single ? 0 : 1], 0.0);
                            EXPECT_EQ(counts[enabled && single ? 1 : 0], 0.0);
                        }
                    }
                }
                lammps_close(lmp);
            }
            EXPECT_EQ(reference.size(), 44 * p);
            equal(reference, candidate, 1.0e-11);
        }
    }
    if (had_previous) setenv(name, saved.c_str(), 1);
    else unsetenv(name);
}
