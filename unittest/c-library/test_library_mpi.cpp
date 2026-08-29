// unit tests for checking LAMMPS configuration settings  through the library interface

#define LAMMPS_LIB_MPI 1
#include "lammps.h"
#include "library.h"
#include "lmptype.h"
#include "timer.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "../testing/test_mpi_main.h"

using ::LAMMPS_NS::tagint;
using ::testing::ExitedWithCode;
using ::testing::HasSubstr;
using ::testing::StartsWith;
using ::testing::StrEq;

TEST(MPI, global_box)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    EXPECT_EQ(nprocs, 4);
    EXPECT_GT(me, -1);
    EXPECT_LT(me, 5);

    double boxlo[3];
    double boxhi[3];
    double xy = 0.0;
    double yz = 0.0;
    double xz = 0.0;
    int pflags[3];
    int boxflag;

    ::testing::internal::CaptureStdout();
    const char *args[] = {"LAMMPS_test", "-log", "none", "-echo", "screen", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
    lammps_command(lmp, "units           lj");
    lammps_command(lmp, "atom_style      atomic");
    lammps_command(lmp, "region          box block 0 2 0 2 0 2");
    lammps_command(lmp, "create_box      1 box");

    lammps_extract_box(lmp, boxlo, boxhi, &xy, &yz, &xz, pflags, &boxflag);
    ::testing::internal::GetCapturedStdout();

    EXPECT_EQ(boxlo[0], 0.0);
    EXPECT_EQ(boxlo[1], 0.0);
    EXPECT_EQ(boxlo[2], 0.0);

    EXPECT_EQ(boxhi[0], 2.0);
    EXPECT_EQ(boxhi[1], 2.0);
    EXPECT_EQ(boxhi[2], 2.0);

    ::testing::internal::CaptureStdout();
    lammps_close(lmp);
    ::testing::internal::GetCapturedStdout();
};

TEST(MPI, sub_box)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    EXPECT_EQ(nprocs, 4);
    EXPECT_GT(me, -1);
    EXPECT_LT(me, 5);

    double boxlo[3];
    double boxhi[3];
    double xy = 0.0;
    double yz = 0.0;
    double xz = 0.0;
    int pflags[3];
    int boxflag;

    ::testing::internal::CaptureStdout();
    const char *args[] = {"LAMMPS_test", "-log", "none", "-echo", "screen", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
    lammps_command(lmp, "units           lj");
    lammps_command(lmp, "atom_style      atomic");
    lammps_command(lmp, "region          box block 0 2 0 2 0 2");
    lammps_command(lmp, "create_box      1 box");

    lammps_extract_box(lmp, boxlo, boxhi, &xy, &yz, &xz, pflags, &boxflag);
    ::testing::internal::GetCapturedStdout();

    EXPECT_EQ(boxlo[0], 0.0);
    EXPECT_EQ(boxlo[1], 0.0);
    EXPECT_EQ(boxlo[2], 0.0);

    EXPECT_EQ(boxhi[0], 2.0);
    EXPECT_EQ(boxhi[1], 2.0);
    EXPECT_EQ(boxhi[2], 2.0);

    auto *sublo = (double *)lammps_extract_global(lmp, "sublo");
    auto *subhi = (double *)lammps_extract_global(lmp, "subhi");

    ASSERT_NE(sublo, nullptr);
    ASSERT_NE(subhi, nullptr);

    EXPECT_GE(sublo[0], boxlo[0]);
    EXPECT_GE(sublo[1], boxlo[1]);
    EXPECT_GE(sublo[2], boxlo[2]);
    EXPECT_LE(subhi[0], boxhi[0]);
    EXPECT_LE(subhi[1], boxhi[1]);
    EXPECT_LE(subhi[2], boxhi[2]);

    ::testing::internal::CaptureStdout();
    lammps_command(lmp, "change_box all triclinic");
    ::testing::internal::GetCapturedStdout();

    sublo = (double *)lammps_extract_global(lmp, "sublo_lambda");
    subhi = (double *)lammps_extract_global(lmp, "subhi_lambda");
    ASSERT_NE(sublo, nullptr);
    ASSERT_NE(subhi, nullptr);

    EXPECT_GE(sublo[0], 0.0);
    EXPECT_GE(sublo[1], 0.0);
    EXPECT_GE(sublo[2], 0.0);
    EXPECT_LE(subhi[0], 1.0);
    EXPECT_LE(subhi[1], 1.0);
    EXPECT_LE(subhi[2], 1.0);

    ::testing::internal::CaptureStdout();
    lammps_close(lmp);
    ::testing::internal::GetCapturedStdout();
};

TEST(MPI, split_comm)
{
    int nprocs, me, color, key;
    MPI_Comm newcomm;
    lammps_mpi_init();
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    color = me % 2;
    key   = me;

    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm);

    const char *args[] = {"LAMMPS_test", "-log", "none", "-echo", "screen", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, newcomm, nullptr);
    lammps_command(lmp, "units           lj");
    lammps_command(lmp, "atom_style      atomic");
    lammps_command(lmp, "region          box block 0 2 0 2 0 2");
    lammps_command(lmp, "create_box      1 box");

    MPI_Comm_size(newcomm, &nprocs);
    MPI_Comm_rank(newcomm, &me);
    EXPECT_EQ(nprocs, 2);
    EXPECT_GT(me, -1);
    EXPECT_LT(me, 2);
    EXPECT_EQ(lammps_extract_setting(lmp, "universe_size"), nprocs);
    EXPECT_EQ(lammps_extract_setting(lmp, "universe_rank"), me);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), nprocs);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_rank"), me);

    lammps_close(lmp);
};

TEST(MPI, multi_partition)
{
    int nprocs, me;
    lammps_mpi_init();
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    const char *args[] = {"LAMMPS_test", "-log",    "none", "-partition", "4x1",  "-echo",
                          "screen",      "-nocite", "-in",  "none",       nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);

    lammps_command(lmp, "units           lj");
    lammps_command(lmp, "atom_style      atomic");
    lammps_command(lmp, "region          box block 0 2 0 2 0 2");
    lammps_command(lmp, "create_box      1 box");
    lammps_command(lmp, "variable        partition universe 1 2 3 4");

    EXPECT_EQ(lammps_extract_setting(lmp, "universe_size"), nprocs);
    EXPECT_EQ(lammps_extract_setting(lmp, "universe_rank"), me);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), 1);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_rank"), 0);

    char *part_id = (char *)lammps_extract_variable(lmp, "partition", nullptr);
    ASSERT_THAT(part_id, StrEq(std::to_string(me + 1)));

    lammps_close(lmp);
};

#if LAMMPS_HAS_PIMD

namespace {

void *open_multirank_partition()
{
    const char *args[] = {"LAMMPS_test", "-screen", "none", "-log",    "none", "-partition",
                          "2x2",         "-in",     "none", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    return lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
}

void create_multirank_two_atom_system(void *lmp)
{
    lammps_command(lmp, "variable x2 world 19.0 10.0");
    lammps_command(lmp, "variable y2 world 10.0 17.0");
    lammps_command(lmp, "units lj");
    lammps_command(lmp, "atom_style atomic");
    lammps_command(lmp, "atom_modify map array");
    lammps_command(lmp, "boundary p p p");
    lammps_command(lmp, "region box block 0.0 20.0 0.0 20.0 0.0 20.0");
    lammps_command(lmp, "create_box 1 box");
    lammps_command(lmp, "create_atoms 1 single 10.0 10.0 2.0");
    lammps_command(lmp, "create_atoms 1 single ${x2} ${y2} 2.0");
    lammps_command(lmp, "mass 1 1.0");
    lammps_command(lmp, "pair_style lj/cut 2.5");
    lammps_command(lmp, "pair_coeff * * 0.0 1.0");
    lammps_command(lmp, "timestep 0.001");
    lammps_command(lmp, "velocity all set 0.0 0.0 0.0");
}

} // namespace

TEST(MPI, pimd_multirank_spring_force)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    ASSERT_EQ(nprocs, 4);

    void *lmp = open_multirank_partition();
    ASSERT_NE(lmp, nullptr);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), 2);
    create_multirank_two_atom_system(lmp);
    lammps_command(lmp, "group first id 1");
    lammps_command(lmp, "group second id 2");
    lammps_command(lmp, "compute first_force first reduce sum fx fy fz");
    lammps_command(lmp, "compute second_force second reduce sum fx fy fz");
    lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nvt "
                        "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
    lammps_command(lmp, "run 0");

    auto *first =
        (double *)lammps_extract_compute(lmp, "first_force", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR);
    auto *second =
        (double *)lammps_extract_compute(lmp, "second_force", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    for (int dimension = 0; dimension < 3; ++dimension)
        EXPECT_NEAR(first[dimension], 0.0, 1.0e-12);
    if (me / 2 == 0) {
        EXPECT_LT(second[0], 0.0);
        EXPECT_GT(second[1], 0.0);
    } else {
        EXPECT_GT(second[0], 0.0);
        EXPECT_LT(second[1], 0.0);
    }
    EXPECT_NEAR(7.0 * second[0] + 9.0 * second[1], 0.0, 1.0e-10);
    EXPECT_NEAR(second[2], 0.0, 1.0e-12);
    lammps_close(lmp);
}

#endif

#if LAMMPS_HAS_PLUMED_PIMD

TEST(MPI, plumed_pimd_input_contract)
{
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    ASSERT_EQ(nprocs, 4);

    auto open_lammps = []() {
        const char *args[] = {"LAMMPS_test", "-screen", "none", "-log",    "none", "-partition",
                              "4x1",         "-in",     "none", "-nocite", nullptr};
        char **argv        = (char **)args;
        int argc           = (sizeof(args) / sizeof(char *)) - 1;
        void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
        lammps_command(lmp, "units lj");
        lammps_command(lmp, "atom_style atomic");
        lammps_command(lmp, "atom_modify map array");
        lammps_command(lmp, "boundary p p p");
        lammps_command(lmp, "region box block 0.0 20.0 0.0 20.0 0.0 20.0");
        lammps_command(lmp, "create_box 1 box");
        lammps_command(lmp, "create_atoms 1 single 10.0 10.0 2.0");
        lammps_command(lmp, "create_atoms 1 single 19.0 10.0 2.0");
        lammps_command(lmp, "mass 1 1.0");
        lammps_command(lmp, "pair_style lj/cut 2.5");
        lammps_command(lmp, "pair_coeff * * 0.0 1.0");
        lammps_command(lmp, "timestep 0.001");
        lammps_command(lmp, "velocity all set 0.0 0.0 0.0");
        lammps_set_show_error(lmp, 0);
        return lmp;
    };
    auto expect_error = [](void *lmp, const char *command, const char *expected_message) {
        lammps_command(lmp, command);
        const int has_error = lammps_has_error(lmp);
        EXPECT_EQ(has_error, 1);
        if (has_error) {
            char error_message[512];
            EXPECT_NE(lammps_get_last_error_message(lmp, error_message, sizeof(error_message)), 0);
            EXPECT_THAT(error_message, HasSubstr(expected_message));
        }
    };

    const std::array<std::array<const char *, 2>, 3> invalid_commands = {{
        {"fix guard all plumed path_integral invalid",
         "Unknown fix plumed path_integral value: invalid"},
        {"fix guard all plumed path_integral centroid",
         "Fix plumed path_integral mode requires the pimd_fix keyword"},
        {"fix guard all plumed pimd_fix fpimd",
         "Fix plumed pimd_fix requires a path_integral mode"},
    }};
    for (const auto &test_case : invalid_commands) {
        void *lmp = open_lammps();
        expect_error(lmp, test_case[0], test_case[1]);
        lammps_close(lmp);
    }

    void *lmp = open_lammps();
    lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nve "
                        "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
    lammps_command(lmp, "fix guard all plumed path_integral centroid pimd_fix fpimd");
    EXPECT_EQ(lammps_has_error(lmp), 0);
    expect_error(lmp, "run 0 post no",
                 "Fix plumed path_integral modes require method pimd and ensemble nvt");
    lammps_close(lmp);
}

TEST(MPI, plumed_pimd_single_bead_limit)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    ASSERT_EQ(nprocs, 4);

    const char *plumed_file = "test_plumed_pimd_single_bead.dat";
    const std::string regular_log =
        "test_plumed_pimd_single_bead_regular." + std::to_string(me) + ".log";
    const std::string centroid_log =
        "test_plumed_pimd_single_bead_centroid." + std::to_string(me) + ".log";
    if (me == 0) {
        std::ofstream restraint(plumed_file);
        restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                  << "bias: RESTRAINT ARG=d AT=6.0 KAPPA=4.0\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const char *args[] = {"LAMMPS_test", "-screen", "none", "-log", "none", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, MPI_COMM_SELF, nullptr);
    ASSERT_NE(lmp, nullptr);
    EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), 1);

    lammps_command(lmp, "units lj");
    lammps_command(lmp, "atom_style atomic");
    lammps_command(lmp, "atom_modify map array");
    lammps_command(lmp, "boundary p p p");
    lammps_command(lmp, "region box block 0.0 20.0 0.0 20.0 0.0 20.0");
    lammps_command(lmp, "create_box 1 box");
    lammps_command(lmp, "create_atoms 1 single 10.0 10.0 2.0");
    lammps_command(lmp, "create_atoms 1 single 19.0 10.0 2.0");
    lammps_command(lmp, "mass 1 1.0");
    lammps_command(lmp, "pair_style lj/cut 2.5");
    lammps_command(lmp, "pair_coeff * * 0.0 1.0");
    lammps_command(lmp, "timestep 0.001");
    lammps_command(lmp, "velocity all set 0.0 0.0 0.0");

    auto run_case = [&](const std::string &fix_command) {
        std::array<double, 7> result{};
        lammps_command(lmp, fix_command.c_str());
        lammps_command(lmp, "run 0 post no");
        // Stock fix plumed exposes its first calculated bias on the next setup.
        lammps_command(lmp, "run 0 post no");
        auto **forces = (double **)lammps_extract_atom(lmp, "f");
        EXPECT_NE(forces, nullptr);
        if (forces) {
            for (int atom = 0; atom < 2; ++atom) {
                const tagint atom_id = atom + 1;
                const int index      = lammps_map_atom(lmp, &atom_id);
                EXPECT_GE(index, 0);
                if (index < 0) continue;
                for (int dimension = 0; dimension < 3; ++dimension)
                    result[3 * atom + dimension] = forces[index][dimension];
            }
        }
        auto *bias =
            (double *)lammps_extract_fix(lmp, "bias", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, -1, -1);
        EXPECT_NE(bias, nullptr);
        if (bias) {
            result[6] = *bias;
            lammps_free(bias);
        }
        lammps_command(lmp, "unfix bias");
        return result;
    };

    const auto regular = run_case("fix bias all plumed plumedfile " + std::string(plumed_file) +
                                  " outfile " + regular_log);
    lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nvt "
                        "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
    const auto centroid =
        run_case("fix bias all plumed plumedfile " + std::string(plumed_file) + " outfile " +
                 centroid_log + " path_integral centroid pimd_fix fpimd");

    EXPECT_NEAR(regular[6], 18.0, 1.0e-12);
    EXPECT_NEAR(centroid[6], 18.0, 1.0e-12);
    for (std::size_t i = 0; i < regular.size(); ++i)
        EXPECT_NEAR(centroid[i], regular[i], 1.0e-12);

    lammps_close(lmp);
    std::remove(regular_log.c_str());
    std::remove(centroid_log.c_str());
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) std::remove(plumed_file);
};

TEST(MPI, plumed_pimd_cyclic_bead_permutation)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    ASSERT_EQ(nprocs, 4);

    const char *centroid_file  = "test_plumed_pimd_cyclic_centroid.dat";
    const char *bead_mean_file = "test_plumed_pimd_cyclic_bead_mean.dat";
    const char *bead_density_file = "test_plumed_pimd_cyclic_bead_density.dat";
    const char *centroid_log   = "test_plumed_pimd_cyclic_centroid.log";
    const char *bead_mean_log  = "test_plumed_pimd_cyclic_bead_mean.log";
    const char *bead_density_log  = "test_plumed_pimd_cyclic_bead_density.log";
    if (me == 0) {
        std::ofstream centroid(centroid_file);
        centroid << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                 << "bias: RESTRAINT ARG=d AT=0.0 KAPPA=4.0\n";
        std::ofstream bead_mean(bead_mean_file);
        bead_mean << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                  << "mean: ENSEMBLE ARG=d\n"
                  << "bias: RESTRAINT ARG=mean.d AT=6.5 KAPPA=4.0\n";
        std::ofstream bead_density(bead_density_file);
        bead_density << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                     << "bias: RESTRAINT ARG=d AT=6.5 KAPPA=4.0\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const std::array<std::array<double, 2>, 4> positions = {
        {{{19.0, 10.0}}, {{10.0, 18.0}}, {{3.0, 10.0}}, {{10.0, 4.0}}}};
    auto run_case = [&](const char *plumed_file, const char *plumed_log,
                        const char *path_integral_mode, int shift) {
        const char *args[] = {"LAMMPS_test", "-screen", "none", "-log",    "none", "-partition",
                              "4x1",         "-in",     "none", "-nocite", nullptr};
        char **argv        = (char **)args;
        int argc           = (sizeof(args) / sizeof(char *)) - 1;
        void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
        EXPECT_NE(lmp, nullptr);
        if (!lmp) return 0.0;

        EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), 1);
        const auto &position = positions[(me + shift) % nprocs];
        lammps_command(lmp, "units lj");
        lammps_command(lmp, "atom_style atomic");
        lammps_command(lmp, "atom_modify map array");
        lammps_command(lmp, "boundary p p p");
        lammps_command(lmp, "region box block 0.0 20.0 0.0 20.0 0.0 20.0");
        lammps_command(lmp, "create_box 1 box");
        lammps_command(lmp, "create_atoms 1 single 10.0 10.0 2.0");
        const std::string create_second = "create_atoms 1 single " + std::to_string(position[0]) +
                                          " " + std::to_string(position[1]) + " 2.0";
        lammps_command(lmp, create_second.c_str());
        lammps_command(lmp, "mass 1 1.0");
        lammps_command(lmp, "pair_style lj/cut 2.5");
        lammps_command(lmp, "pair_coeff * * 0.0 1.0");
        lammps_command(lmp, "timestep 0.001");
        lammps_command(lmp, "velocity all set 0.0 0.0 0.0");
        lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nvt "
                            "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
        const std::string fix_command = "fix bias all plumed plumedfile " +
                                        std::string(plumed_file) + " outfile " + plumed_log + "." +
                                        std::to_string(shift) + " path_integral " +
                                        path_integral_mode + " pimd_fix fpimd";
        lammps_command(lmp, fix_command.c_str());
        lammps_command(lmp, "run 0 post no");

        double result = 0.0;
        auto *bias =
            (double *)lammps_extract_fix(lmp, "bias", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, -1, -1);
        EXPECT_NE(bias, nullptr);
        if (bias) {
            result = *bias;
            lammps_free(bias);
        }
        lammps_close(lmp);
        return result;
    };

    const double centroid_original  = run_case(centroid_file, centroid_log, "centroid", 0);
    const double centroid_shifted   = run_case(centroid_file, centroid_log, "centroid", 1);
    const double bead_mean_original = run_case(bead_mean_file, bead_mean_log, "bead_mean", 0);
    const double bead_mean_shifted  = run_case(bead_mean_file, bead_mean_log, "bead_mean", 1);
    const double bead_density_original =
        run_case(bead_density_file, bead_density_log, "bead_density", 0);
    const double bead_density_shifted =
        run_case(bead_density_file, bead_density_log, "bead_density", 1);
    EXPECT_NEAR(centroid_original, centroid_shifted, 1.0e-12);
    EXPECT_NEAR(bead_mean_original, bead_mean_shifted, 1.0e-12);
    EXPECT_NEAR(bead_density_original, bead_density_shifted, 1.0e-12);
    EXPECT_NEAR(centroid_original, me == 0 ? 1.0 : 0.0, 1.0e-12);
    EXPECT_NEAR(bead_mean_original, me == 0 ? 2.0 : 0.0, 1.0e-12);
    EXPECT_NEAR(bead_density_original, me == 0 ? 4.5 : 0.0, 1.0e-12);

    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) {
        std::remove(centroid_file);
        std::remove(bead_mean_file);
        std::remove(bead_density_file);
        std::remove((std::string(centroid_log) + ".0").c_str());
        std::remove((std::string(centroid_log) + ".1").c_str());
    }
    for (int shift = 0; shift < 2; ++shift)
        std::remove(
            (std::string(bead_mean_log) + "." + std::to_string(shift) + "." + std::to_string(me))
                .c_str());
    for (int shift = 0; shift < 2; ++shift)
        std::remove(
            (std::string(bead_density_log) + "." + std::to_string(shift) + "." + std::to_string(me))
                .c_str());
};

TEST(MPI, plumed_pimd_bias_modes)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    ASSERT_EQ(nprocs, 4);

    const char *bead_zero_file          = "test_plumed_pimd_bead_zero.dat";
    const char *bead_restraint_file     = "test_plumed_pimd_bead_restraint.dat";
    const char *centroid_zero_file      = "test_plumed_pimd_centroid_zero.dat";
    const char *centroid_restraint_file = "test_plumed_pimd_centroid_restraint.dat";
    const char *density_zero_file       = "test_plumed_pimd_density_zero.dat";
    const char *density_restraint_file  = "test_plumed_pimd_density_restraint.dat";
    const char *bead_zero_log           = "test_plumed_pimd_bead_zero.log";
    const char *bead_restraint_log      = "test_plumed_pimd_bead_restraint.log";
    const char *centroid_zero_log       = "test_plumed_pimd_centroid_zero.log";
    const char *centroid_restraint_log  = "test_plumed_pimd_centroid_restraint.log";
    const char *density_zero_log        = "test_plumed_pimd_density_zero.log";
    const char *density_restraint_log   = "test_plumed_pimd_density_restraint.log";
    if (me == 0) {
        std::ofstream bead_zero(bead_zero_file);
        bead_zero << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                  << "mean: ENSEMBLE ARG=d\n";
        std::ofstream bead_restraint(bead_restraint_file);
        bead_restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                       << "mean: ENSEMBLE ARG=d\n"
                       << "bias: RESTRAINT ARG=mean.d AT=6.5 KAPPA=4.0\n";
        std::ofstream centroid_zero(centroid_zero_file);
        centroid_zero << "d: DISTANCE ATOMS=1,2 NOPBC\n";
        std::ofstream centroid_restraint(centroid_restraint_file);
        centroid_restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                           << "bias: RESTRAINT ARG=d AT=0.0 KAPPA=4.0\n";
        std::ofstream density_zero(density_zero_file);
        density_zero << "d: DISTANCE ATOMS=1,2 NOPBC\n";
        std::ofstream density_restraint(density_restraint_file);
        density_restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                          << "bias: RESTRAINT ARG=d AT=6.5 KAPPA=4.0\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const char *args[] = {"LAMMPS_test", "-screen", "none", "-log",    "none", "-partition",
                          "4x1",         "-in",     "none", "-nocite", nullptr};
    char **argv        = (char **)args;
    int argc           = (sizeof(args) / sizeof(char *)) - 1;
    void *lmp          = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
    ASSERT_NE(lmp, nullptr);

    lammps_command(lmp, "variable x2 world 19.0 10.0 3.0 10.0");
    lammps_command(lmp, "variable y2 world 10.0 18.0 10.0 4.0");
    lammps_command(lmp, "units lj");
    lammps_command(lmp, "atom_style atomic");
    lammps_command(lmp, "atom_modify map array");
    lammps_command(lmp, "boundary p p p");
    lammps_command(lmp, "region box block 0.0 20.0 0.0 20.0 0.0 20.0");
    lammps_command(lmp, "create_box 1 box");
    lammps_command(lmp, "create_atoms 1 single 10.0 10.0 2.0");
    lammps_command(lmp, "create_atoms 1 single ${x2} ${y2} 2.0");
    lammps_command(lmp, "mass 1 1.0");
    lammps_command(lmp, "pair_style lj/cut 2.5");
    lammps_command(lmp, "pair_coeff * * 0.0 1.0");
    lammps_command(lmp, "timestep 0.001");
    lammps_command(lmp, "velocity all set 0.0 0.0 0.0");
    lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nvt "
                        "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
    const std::array<std::array<double, 6>, 4> bead_force_delta = {
        {{1.0, 0.0, 0.0, -1.0, 0.0, 0.0},
         {0.0, 1.0, 0.0, 0.0, -1.0, 0.0},
         {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0},
         {0.0, -1.0, 0.0, 0.0, 1.0, 0.0}}};
    const std::array<double, 6> centroid_force_delta = {0.5, 0.5, 0.0, -0.5, -0.5, 0.0};
    const std::array<std::array<double, 6>, 4> density_force_delta = {
        {{2.5, 0.0, 0.0, -2.5, 0.0, 0.0},
         {0.0, 1.5, 0.0, 0.0, -1.5, 0.0},
         {-0.5, 0.0, 0.0, 0.5, 0.0, 0.0},
         {0.0, 0.5, 0.0, 0.0, -0.5, 0.0}}};
    auto extract_forces                              = [&]() {
        std::array<double, 6> result{};
        auto **forces = (double **)lammps_extract_atom(lmp, "f");
        EXPECT_NE(forces, nullptr);
        if (!forces) return result;

        for (int atom = 0; atom < 2; ++atom) {
            const tagint atom_id = atom + 1;
            const int index      = lammps_map_atom(lmp, &atom_id);
            EXPECT_GE(index, 0);
            if (index < 0) continue;
            for (int dimension = 0; dimension < 3; ++dimension)
                result[3 * atom + dimension] = forces[index][dimension];
        }
        return result;
    };
    auto check_mode = [&](const char *zero_command, const char *bias_command, double expected_bias,
                          const std::array<double, 6> &expected_force_delta) {
        lammps_command(lmp, zero_command);
        lammps_command(lmp, "run 0 post no");
        const auto zero_forces = extract_forces();
        lammps_command(lmp, "unfix zero");

        lammps_command(lmp, bias_command);
        lammps_command(lmp, "run 0 post no");
        const auto biased_forces = extract_forces();
        auto *bias =
            (double *)lammps_extract_fix(lmp, "bias", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, -1, -1);
        ASSERT_NE(bias, nullptr);
        EXPECT_NEAR(*bias, me == 0 ? expected_bias : 0.0, 1.0e-12);
        lammps_free(bias);

        for (std::size_t i = 0; i < expected_force_delta.size(); ++i)
            EXPECT_NEAR(biased_forces[i] - zero_forces[i], expected_force_delta[i], 1.0e-12);
        lammps_command(lmp, "unfix bias");
    };

    check_mode("fix zero all plumed plumedfile test_plumed_pimd_bead_zero.dat "
               "outfile test_plumed_pimd_bead_zero.log path_integral bead_mean pimd_fix fpimd",
               "fix bias all plumed plumedfile test_plumed_pimd_bead_restraint.dat "
               "outfile test_plumed_pimd_bead_restraint.log path_integral bead_mean pimd_fix "
               "fpimd",
               2.0, bead_force_delta[me]);
    check_mode("fix zero all plumed plumedfile test_plumed_pimd_centroid_zero.dat "
               "outfile test_plumed_pimd_centroid_zero.log path_integral centroid pimd_fix fpimd",
               "fix bias all plumed plumedfile test_plumed_pimd_centroid_restraint.dat "
               "outfile test_plumed_pimd_centroid_restraint.log path_integral centroid pimd_fix "
               "fpimd",
               1.0, centroid_force_delta);
    check_mode("fix zero all plumed plumedfile test_plumed_pimd_density_zero.dat "
               "outfile test_plumed_pimd_density_zero.log path_integral bead_density pimd_fix "
               "fpimd",
               "fix bias all plumed plumedfile test_plumed_pimd_density_restraint.dat "
               "outfile test_plumed_pimd_density_restraint.log path_integral bead_density "
               "pimd_fix fpimd",
               4.5, density_force_delta[me]);

    lammps_close(lmp);
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) {
        std::remove(bead_zero_file);
        std::remove(bead_restraint_file);
        std::remove(centroid_zero_file);
        std::remove(centroid_restraint_file);
        std::remove(density_zero_file);
        std::remove(density_restraint_file);
        for (int i = 0; i < nprocs; ++i) {
            std::remove((std::string(bead_zero_log) + "." + std::to_string(i)).c_str());
            std::remove((std::string(bead_restraint_log) + "." + std::to_string(i)).c_str());
        }
        std::remove(centroid_zero_log);
        std::remove(centroid_restraint_log);
    }
    std::remove((std::string(density_zero_log) + "." + std::to_string(me)).c_str());
    std::remove((std::string(density_restraint_log) + "." + std::to_string(me)).c_str());
};

TEST(MPI, plumed_pimd_multirank_bead_mean)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    ASSERT_EQ(nprocs, 4);

    const char *zero_file              = "test_plumed_pimd_multirank_zero.dat";
    const char *restraint_file         = "test_plumed_pimd_multirank_restraint.dat";
    const char *density_zero_file      = "test_plumed_pimd_multirank_density_zero.dat";
    const char *density_restraint_file = "test_plumed_pimd_multirank_density_restraint.dat";
    const char *zero_log               = "test_plumed_pimd_multirank_zero.log";
    const char *restraint_log          = "test_plumed_pimd_multirank_restraint.log";
    const char *density_zero_log       = "test_plumed_pimd_multirank_density_zero.log";
    const char *density_restraint_log  = "test_plumed_pimd_multirank_density_restraint.log";
    if (me == 0) {
        std::ofstream zero(zero_file);
        zero << "d: DISTANCE ATOMS=1,2 NOPBC\n"
             << "mean: ENSEMBLE ARG=d\n";
        std::ofstream restraint(restraint_file);
        restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                  << "mean: ENSEMBLE ARG=d\n"
                  << "bias: RESTRAINT ARG=mean.d AT=7.5 KAPPA=4.0\n";
        std::ofstream density_zero(density_zero_file);
        density_zero << "d: DISTANCE ATOMS=1,2 NOPBC\n";
        std::ofstream density_restraint(density_restraint_file);
        density_restraint << "d: DISTANCE ATOMS=1,2 NOPBC\n"
                          << "bias: RESTRAINT ARG=d AT=7.5 KAPPA=4.0\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int bead = me / 2;
    ASSERT_GE(bead, 0);
    ASSERT_LT(bead, 2);
    auto run_case = [&](const char *plumed_file, const char *plumed_log, const char *mode) {
        std::array<double, 7> result{};
        void *lmp = open_multirank_partition();
        EXPECT_NE(lmp, nullptr);
        if (!lmp) return result;

        EXPECT_EQ(lammps_extract_setting(lmp, "world_size"), 2);
        create_multirank_two_atom_system(lmp);
        auto *nlocal = (int *)lammps_extract_global(lmp, "nlocal");
        EXPECT_NE(nlocal, nullptr);
        int zero_atom_ranks = nlocal && *nlocal == 0;
        MPI_Allreduce(MPI_IN_PLACE, &zero_atom_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_GT(zero_atom_ranks, 0);
        lammps_command(lmp, "group first id 1");
        lammps_command(lmp, "group second id 2");
        lammps_command(lmp, "compute first_force first reduce sum fx fy fz");
        lammps_command(lmp, "compute second_force second reduce sum fx fy fz");
        lammps_command(lmp, "fix fpimd all pimd/langevin method pimd ensemble nvt "
                            "integrator obabo thermostat PILE_L 1234 tau 1.0 temp 1.0 fixcom no");
        const std::string fix_command = "fix bias all plumed plumedfile " +
                                        std::string(plumed_file) + " outfile " + plumed_log +
                                        " path_integral " + mode + " pimd_fix fpimd";
        lammps_command(lmp, fix_command.c_str());
        lammps_command(lmp, "run 0");

        auto *first =
            (double *)lammps_extract_compute(lmp, "first_force", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR);
        auto *second = (double *)lammps_extract_compute(lmp, "second_force", LMP_STYLE_GLOBAL,
                                                        LMP_TYPE_VECTOR);
        EXPECT_NE(first, nullptr);
        EXPECT_NE(second, nullptr);
        if (first && second) {
            for (int dimension = 0; dimension < 3; ++dimension) {
                result[dimension]     = first[dimension];
                result[3 + dimension] = second[dimension];
            }
        }
        auto *bias =
            (double *)lammps_extract_fix(lmp, "bias", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, -1, -1);
        EXPECT_NE(bias, nullptr);
        if (bias) {
            result[6] = *bias;
            lammps_free(bias);
        }
        lammps_close(lmp);
        return result;
    };

    const auto zero_result   = run_case(zero_file, zero_log, "bead_mean");
    const auto biased_result = run_case(restraint_file, restraint_log, "bead_mean");
    EXPECT_NEAR(zero_result[6], 0.0, 1.0e-12);
    EXPECT_NEAR(biased_result[6], bead == 0 ? 0.5 : 0.0, 1.0e-12);

    const std::array<std::array<double, 6>, 2> expected_force_delta = {
        {{1.0, 0.0, 0.0, -1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0, -1.0, 0.0}}};
    for (std::size_t i = 0; i < expected_force_delta[bead].size(); ++i)
        EXPECT_NEAR(biased_result[i] - zero_result[i], expected_force_delta[bead][i], 1.0e-12);

    const auto density_zero_result = run_case(density_zero_file, density_zero_log, "bead_density");
    const auto density_biased_result =
        run_case(density_restraint_file, density_restraint_log, "bead_density");
    EXPECT_NEAR(density_zero_result[6], 0.0, 1.0e-12);
    EXPECT_NEAR(density_biased_result[6], bead == 0 ? 2.5 : 0.0, 1.0e-12);

    const std::array<std::array<double, 6>, 2> expected_density_force_delta = {
        {{3.0, 0.0, 0.0, -3.0, 0.0, 0.0}, {0.0, -1.0, 0.0, 0.0, 1.0, 0.0}}};
    for (std::size_t i = 0; i < expected_density_force_delta[bead].size(); ++i)
        EXPECT_NEAR(density_biased_result[i] - density_zero_result[i],
                    expected_density_force_delta[bead][i], 1.0e-12);

    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) {
        std::remove(zero_file);
        std::remove(restraint_file);
        std::remove(density_zero_file);
        std::remove(density_restraint_file);
        for (int bead_index = 0; bead_index < 2; ++bead_index) {
            std::remove((std::string(zero_log) + "." + std::to_string(bead_index)).c_str());
            std::remove((std::string(restraint_log) + "." + std::to_string(bead_index)).c_str());
            std::remove((std::string(density_zero_log) + "." + std::to_string(bead_index)).c_str());
            std::remove(
                (std::string(density_restraint_log) + "." + std::to_string(bead_index)).c_str());
        }
    }
};

#endif

class MPITest : public ::testing::Test {
public:
    void command(const std::string &line) { lammps_command(lmp, line.c_str()); }

protected:
    const char *testbinary = "LAMMPSTest";
    void *lmp;

    void SetUp() override
    {
        const char *args[] = {testbinary, "-log", "none", "-echo", "screen", "-nocite", nullptr};
        char **argv        = (char **)args;
        int argc           = (sizeof(args) / sizeof(char *)) - 1;
        if (!verbose) ::testing::internal::CaptureStdout();
        lmp = lammps_open(argc, argv, MPI_COMM_WORLD, nullptr);
        InitSystem();
        if (!verbose) ::testing::internal::GetCapturedStdout();
    }

    virtual void InitSystem()
    {
        command("units           lj");
        command("atom_style      atomic");
        command("atom_modify     map yes");

        command("lattice         fcc 0.8442");
        command("region          box block 0 2 0 2 0 2");
        command("create_box      1 box");
        command("create_atoms    1 box");
        command("mass            1 1.0");

        command("velocity        all create 3.0 87287");

        command("pair_style      lj/cut 2.5");
        command("pair_coeff      1 1 1.0 1.0 2.5");

        command("neighbor        0.3 bin");
        command("neigh_modify    every 20 delay 0 check no");
    }

    void TearDown() override
    {
        if (!verbose) ::testing::internal::CaptureStdout();
        lammps_close(lmp);
        lmp = nullptr;
        if (!verbose) ::testing::internal::GetCapturedStdout();
    }
};

TEST_F(MPITest, size_rank)
{
    int nprocs, me;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    EXPECT_EQ(nprocs, lammps_extract_setting(lmp, "world_size"));
    EXPECT_EQ(me, lammps_extract_setting(lmp, "world_rank"));
}

#if !defined(LAMMPS_BIGBIG)

TEST_F(MPITest, gather)
{
    auto natoms = (int64_t)lammps_get_natoms(lmp);
    ASSERT_EQ(natoms, 32);
    int *p_nlocal = (int *)lammps_extract_global(lmp, "nlocal");
    int nlocal    = *p_nlocal;
    EXPECT_LT(nlocal, 32);
    EXPECT_EQ(nlocal, 8);

    // get the entire x on all procs
    auto *x = new double[natoms * 3];
    lammps_gather(lmp, (char *)"x", 1, 3, x);

    int *tag       = (int *)lammps_extract_atom(lmp, "id");
    auto **x_local = (double **)lammps_extract_atom(lmp, "x");

    // each proc checks its local atoms
    for (int i = 0; i < nlocal; i++) {
        int64_t j   = tag[i] - 1;
        double *x_i = x_local[i];
        double *x_g = &x[j * 3];
        EXPECT_DOUBLE_EQ(x_g[0], x_i[0]);
        EXPECT_DOUBLE_EQ(x_g[1], x_i[1]);
        EXPECT_DOUBLE_EQ(x_g[2], x_i[2]);
    }

    delete[] x;
}

TEST_F(MPITest, scatter)
{
    int *p_nlocal  = (int *)lammps_extract_global(lmp, "nlocal");
    int nlocal     = *p_nlocal;
    auto *x_orig   = new double[3 * nlocal];
    auto **x_local = (double **)lammps_extract_atom(lmp, "x");

    // make copy of original local x vector
    for (int i = 0; i < nlocal; i++) {
        int j         = 3 * i;
        x_orig[j]     = x_local[i][0];
        x_orig[j + 1] = x_local[i][1];
        x_orig[j + 2] = x_local[i][2];
    }

    // get the entire x on all procs
    auto natoms = (int64_t)lammps_get_natoms(lmp);
    auto *x     = new double[natoms * 3];
    lammps_gather(lmp, (char *)"x", 1, 3, x);

    // shift all coordinates by 0.001
    const double delta = 0.001;
    for (int64_t i = 0; i < 3 * natoms; i++)
        x[i] += delta;

    // update positions of all atoms
    lammps_scatter(lmp, (char *)"x", 1, 3, x);
    delete[] x;
    x = nullptr;

    // get new nlocal and x_local
    p_nlocal = (int *)lammps_extract_global(lmp, "nlocal");
    nlocal   = *p_nlocal;
    x_local  = (double **)lammps_extract_atom(lmp, "x");

    ASSERT_EQ(nlocal, 8);

    // each proc checks its local atoms for shift
    for (int i = 0; i < nlocal; i++) {
        double *x_a = x_local[i];
        double *x_b = &x_orig[i * 3];
        EXPECT_DOUBLE_EQ(x_a[0], x_b[0] + delta);
        EXPECT_DOUBLE_EQ(x_a[1], x_b[1] + delta);
        EXPECT_DOUBLE_EQ(x_a[2], x_b[2] + delta);
    }

    delete[] x_orig;
}
#endif
