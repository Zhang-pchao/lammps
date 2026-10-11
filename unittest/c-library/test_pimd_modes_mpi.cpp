/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Analytical eigenvalue and mass checks for the legacy NVT normal-mode convention.
#define LAMMPS_LIB_MPI 1
#include "REPLICA/fix_pimd_nvt.h"
#include "creator_registry.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"

#include <cmath>
#include <string>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

namespace {
class ModeProbe : public LAMMPS_NS::FixPIMDNVT {
public:
    ModeProbe(LAMMPS_NS::LAMMPS *lmp, int narg, char **arg) : FixPIMDNVT(lmp, narg, arg) {}
    double eigenvalue(int mode) const { return lam[mode]; }
    double type_mass() const { return mass[1]; }
    double forward_centroid(int bead) const { return M_x2xp[0][bead]; }
    double backward_centroid(int bead) const { return M_xp2x[bead][0]; }
};

LAMMPS_NS::Fix *create_probe(LAMMPS_NS::LAMMPS *lmp, int narg, char **arg)
{
    return new ModeProbe(lmp, narg, arg);
}

void check_modes(int beads)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, beads);
    MPI_Comm subset = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < beads ? 0 : MPI_UNDEFINED, rank, &subset);
    if (subset == MPI_COMM_NULL) return;

    std::string partition = std::to_string(beads) + "x1";
    const char *args[]    = {"LAMMPS_test", "-screen",         "none", "-log", "none",
                             "-partition",  partition.c_str(), "-in",  "none", "-nocite"};
    auto *lmp             = static_cast<LAMMPS_NS::LAMMPS *>(
        lammps_open(10, const_cast<char **>(args), subset, nullptr));
    LAMMPS_NS::Modify::fix_styles().set_plugin("pimd/mode/probe", &create_probe);
    lammps_commands_string(
        lmp, "units real\n"
             "atom_style atomic\n"
             "atom_modify map array\n"
             "region box block 0 4 0 4 0 4\n"
             "create_box 1 box\n"
             "create_atoms 1 single 2 2 2\n"
             "mass 1 2\n"
             "pair_style zero 1\n"
             "pair_coeff * *\n"
             "fix probe all pimd/mode/probe method nmpimd temp 1 fmass 0.5 sp 1 nhc 2\n"
             "run 0\n");
    auto *probe = static_cast<ModeProbe *>(lmp->modify->get_fix_by_id("probe"));
    EXPECT_NE(probe, nullptr);
    if (probe) {
        const double pi = std::acos(-1.0);
        EXPECT_DOUBLE_EQ(probe->eigenvalue(0), 0.0);
        for (int mode = 1; mode < beads; ++mode) {
            const int wave        = (mode + 1) / 2;
            const double sine     = std::sin(pi * wave / beads);
            const double expected = 4.0 * beads * sine * sine;
            SCOPED_TRACE(mode);
            EXPECT_NEAR(probe->eigenvalue(mode), expected, 1.0e-12);
        }
        const int wave             = (rank + 1) / 2;
        const double sine          = std::sin(pi * wave / beads);
        const double expected_mass = rank == 0 ? 2.0 : 2.0 * 0.5 * 4.0 * beads * sine * sine;
        EXPECT_NEAR(probe->type_mass(), expected_mass, 1.0e-12);
        for (int bead = 0; bead < beads; ++bead) {
            EXPECT_DOUBLE_EQ(probe->forward_centroid(bead), 1.0 / beads);
            EXPECT_DOUBLE_EQ(probe->backward_centroid(bead), 1.0);
        }
    }
    lammps_close(lmp);
    LAMMPS_NS::Modify::fix_styles().clear_plugin("pimd/mode/probe");
    MPI_Comm_free(&subset);
}
} // namespace

TEST(PIMDModes, nvt_two_beads)
{
    check_modes(2);
}
TEST(PIMDModes, nvt_three_beads)
{
    check_modes(3);
}
TEST(PIMDModes, nvt_five_beads)
{
    check_modes(5);
}
