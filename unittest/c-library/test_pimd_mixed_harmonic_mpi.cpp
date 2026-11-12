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
#include "error.h"
#include "fix.h"
#include "force.h"
#include "lammps.h"
#include "library.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "safe_pointers.h"
#include "universe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;

namespace {
// Test-only, nonperiodic trapped particles. No mixed-map or mode helper is
// used here. Trap stiffnesses and masses are 1,2,3; pair couplings 12,13,23
// are 0.7,0.2,0.4. The two-particle controls omit particle 3 and its couplings.
class HarmonicFixture : public Pair {
public:
    explicit HarmonicFixture(LAMMPS *lmp) : Pair(lmp)
    {
        single_enable = restartinfo = 0;
        no_virial_fdotr_compute     = 1;
    }
    ~HarmonicFixture() override
    {
        if (allocated) {
            memory->destroy(setflag);
            memory->destroy(cutsq);
        }
    }
    void settings(int narg, char **) override
    {
        if (narg) error->all(FLERR, "Harmonic test fixture takes no options");
    }
    void coeff(int narg, char **) override
    {
        if (narg != 2) error->all(FLERR, "Harmonic test fixture requires pair_coeff * *");
        if (!allocated) {
            allocated = 1;
            memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "harmonic_test:setflag");
            memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "harmonic_test:cutsq");
        }
        for (int i = 1; i <= atom->ntypes; ++i)
            for (int j = 1; j <= atom->ntypes; ++j)
                setflag[i][j] = 1;
    }
    void init_style() override
    {
        if (((atom->natoms != 2) && (atom->natoms != 3)) || !atom->tag_consecutive())
            error->all(FLERR, "Harmonic test fixture requires two or three consecutive atom IDs");
    }
    double init_one(int, int) override { return 2.0; }
    void compute(int eflag, int vflag) override
    {
        ev_init(eflag, vflag);
        const int particles = atom->natoms;
        double q[9]         = {};
        for (int i = 0; i < atom->nlocal; ++i)
            domain->unmap(atom->x[i], atom->image[i], q + 3 * (atom->tag[i] - 1));
        MPI_Allreduce(MPI_IN_PLACE, q, 3 * particles, MPI_DOUBLE, MPI_SUM, world);
        const double coupling[3][3] = {{0.0, 0.7, 0.2}, {0.7, 0.0, 0.4}, {0.2, 0.4, 0.0}};
        for (int i = 0; i < atom->nlocal; ++i) {
            const int species = atom->tag[i] - 1;
            double energy     = 0.0, f[3];
            const double *x   = q + 3 * species;
            for (int d = 0; d < 3; ++d) {
                f[d] = -(species + 1) * x[d];
                energy += 0.5 * (species + 1) * x[d] * x[d];
                for (int other = 0; other < particles; ++other) {
                    if (other == species) continue;
                    const double delta = x[d] - q[3 * other + d];
                    f[d] -= coupling[species][other] * delta;
                    energy += 0.25 * coupling[species][other] * delta * delta;
                }
                atom->f[i][d] += f[d];
            }
            if (eflag_global) eng_vdwl += energy;
            if (eflag_atom) eatom[i] += energy;
            const double stress[6] = {x[0] * f[0], x[1] * f[1], x[2] * f[2],
                                      x[0] * f[1], x[0] * f[2], x[1] * f[2]};
            for (int k = 0; k < 6; ++k) {
                if (vflag_global) virial[k] += stress[k];
                if (vflag_atom) vatom[i][k] += stress[k];
            }
        }
    }
};

Pair *create_fixture(LAMMPS *lmp)
{
    return new HarmonicFixture(lmp);
}

struct FixtureRegistration {
    FixtureRegistration()
    {
        Force::pair_styles().set_plugin("mixed/harmonic/test", create_fixture);
    }
    ~FixtureRegistration() { Force::pair_styles().clear_plugin("mixed/harmonic/test"); }
};

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

bool prepare(LAMMPS *lmp, int particles = 2)
{
    std::string input =
        "units lj\natom_style atomic\nboundary f f f\natom_modify map array sort 0 0\n"
        "region box block -16 16 -16 16 -16 16\ncreate_box " +
        std::to_string(particles) +
        " box\n"
        "create_atoms 1 single 0.5 0.3 -0.2\ncreate_atoms 2 single -0.4 0.1 0.2\n"
        "mass 1 1\nmass 2 2\n";
    if (particles == 3) input += "create_atoms 3 single 0.2 -0.5 0.3\nmass 3 3\ngroup three id 3\n";
    input +=
        "pair_style mixed/harmonic/test\npair_coeff * *\n"
        "group one id 1\ngroup two id 2\nneighbor 0.3 bin\nneigh_modify every 1 delay 0 check no\n"
        "velocity all set 0 0 0\ntimestep 0.002\nthermo 100000000";
    return commands(lmp, input);
}

double energy(LAMMPS *lmp)
{
    const double local = lmp->force->pair->eng_vdwl;
    double total;
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    return total;
}

std::vector<double> sample(LAMMPS *lmp, int p1, int p2, int p3)
{
    const int maximum = std::max({p1, p2, p3}), b = lmp->universe->iworld, n = p1 + p2 + p3;
    std::vector<double> frame(6 * n + 5, 0.0);
    for (int i = 0; i < lmp->atom->nlocal; ++i) {
        const int species = lmp->atom->tag[i] - 1;
        const int p       = species == 0 ? p1 : (species == 1 ? p2 : p3);
        if (b % (maximum / p)) continue;
        const int index = (species == 0 ? 0 : (species == 1 ? p1 : p1 + p2)) + b / (maximum / p);
        double x[3];
        lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], x);
        for (int d = 0; d < 3; ++d) {
            frame[6 * index + d]     = x[d];
            frame[6 * index + 3 + d] = lmp->atom->v[i][d];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, frame.data(), 6 * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto *fix = lmp->modify->get_fix_by_id("pi");
    for (int k = 0; k < 5; ++k)
        frame[6 * n + k] = fix->compute_vector(k < 3 ? k : k + 1);
    return frame;
}

void generate_samples(int p1, int p2, int p3 = 0)
{
    FixtureRegistration registration;
    const bool extended  = std::getenv("LAMMPS_MIXED_HARMONIC_LONG") != nullptr;
    const int warmup     = extended ? 50000 : 50;
    const int production = extended ? 1000000 : 100;
    constexpr int stride = 10;
    for (const int seed : {19771, 39133, 59209, 79187}) {
        auto *lmp = open_system(std::max({p1, p2, p3}));
        if (prepare(lmp, p3 ? 3 : 2) &&
            commands(lmp, "fix pi all pimd/mixed temp 1 sp 6.283185307179586 beads one " +
                              std::to_string(p1) + " beads two " + std::to_string(p2) +
                              (p3 ? " beads three " + std::to_string(p3) : "") +
                              " thermostat pile 1 " + std::to_string(seed)) &&
            commands(lmp, "run " + std::to_string(warmup))) {
            const std::string file = "mixed_harmonic_" + std::to_string(p1) + "_" +
                                     std::to_string(p2) + (p3 ? "_" + std::to_string(p3) : "") +
                                     "_" + std::to_string(seed) + ".bin";
            SafeFilePtr output;
            int bad = 0;
            if (lmp->universe->me == 0) {
                output = std::fopen(file.c_str(), "wb");
                if (!output)
                    bad = 1;
                else {
                    const uint64_t header[5] = {
                        static_cast<uint64_t>(p1 + p2 + p3),
                        static_cast<uint64_t>(production / stride), static_cast<uint64_t>(p1),
                        static_cast<uint64_t>(p2), static_cast<uint64_t>(p3)};
                    const double times[2] = {0.002, 0.002 * stride};
                    bad |= std::fwrite(p3 ? "PIMDMS02" : "PIMDMS01", 1, 8, output) != 8;
                    bad |=
                        std::fwrite(header, sizeof(uint64_t), p3 ? 5 : 4, output) != (p3 ? 5u : 4u);
                    bad |= std::fwrite(times, sizeof(double), 2, output) != 2;
                }
            }
            MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
            EXPECT_EQ(bad, 0);
            if (!bad) {
                for (int step = 0; step < production; step += stride) {
                    if (!commands(lmp, "run 10 pre no post no")) break;
                    const auto frame = sample(lmp, p1, p2, p3);
                    EXPECT_TRUE(std::all_of(frame.begin(), frame.end(), [](double value) {
                        return std::isfinite(value);
                    }));
                    if (lmp->universe->me == 0)
                        bad |= std::fwrite(frame.data(), sizeof(double), frame.size(), output) !=
                               frame.size();
                }
                if (lmp->universe->me == 0) bad |= std::fflush(output) != 0;
                MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
                EXPECT_EQ(bad, 0);
            }
            output = nullptr;
            if (!extended && lmp->universe->me == 0) std::remove(file.c_str());
        }
        lammps_close(lmp);
    }
}
} // namespace

namespace {
void fixture_force_finite_differences(int particles)
{
    FixtureRegistration registration;
    auto *lmp = open_system(1);
    if (prepare(lmp, particles) && commands(lmp, "fix pi all nve\nrun 0")) {
        for (int tag = 1; tag <= particles; ++tag) {
            for (int d = 0; d < 3; ++d) {
                double analytic = 0;
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    if (lmp->atom->tag[i] == tag) analytic = lmp->atom->f[i][d];
                MPI_Allreduce(MPI_IN_PLACE, &analytic, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
                double values[2] = {};
                for (int side = 0; side < 2; ++side) {
                    for (int i = 0; i < lmp->atom->nlocal; ++i)
                        if (lmp->atom->tag[i] == tag)
                            lmp->atom->x[i][d] += (side ? -2 : 1) * 1.0e-6;
                    commands(lmp, "run 0");
                    values[side] = energy(lmp);
                }
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    if (lmp->atom->tag[i] == tag) lmp->atom->x[i][d] += 1.0e-6;
                commands(lmp, "run 0");
                EXPECT_NEAR(analytic, -(values[0] - values[1]) / 2.0e-6, 1.0e-8);
            }
        }
    }
    lammps_close(lmp);
}

} // namespace

TEST(PIMDMixedHarmonic, fixture_force_finite_differences)
{
    fixture_force_finite_differences(2);
}
TEST(PIMDMixedHarmonic, three_particle_fixture_force_finite_differences)
{
    fixture_force_finite_differences(3);
}
TEST(PIMDMixedHarmonic, three_species_samples)
{
    generate_samples(8, 2, 1);
}

TEST(PIMDMixedHarmonic, classical_samples)
{
    generate_samples(1, 1);
}
TEST(PIMDMixedHarmonic, mixed_samples)
{
    generate_samples(4, 1);
}
TEST(PIMDMixedHarmonic, nested_samples)
{
    generate_samples(4, 2);
}
TEST(PIMDMixedHarmonic, uniform_samples)
{
    generate_samples(4, 4);
}

TEST(PIMDMixedHarmonic, fixture_virial_finite_differences)
{
    FixtureRegistration registration;
    auto *lmp = open_system(1);
    if (prepare(lmp) && commands(lmp, "fix pi all nve\nrun 0")) {
        double stress[6];
        MPI_Allreduce(lmp->force->pair->virial, stress, 6, MPI_DOUBLE, MPI_SUM, lmp->world);
        const int tensor[3][3]         = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
        const double coordinates[2][3] = {{0.5, 0.3, -0.2}, {-0.4, 0.1, 0.2}};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double values[2] = {};
                for (int side = 0; side < 2; ++side) {
                    for (int i = 0; i < lmp->atom->nlocal; ++i) {
                        const int tag = lmp->atom->tag[i] - 1;
                        lmp->atom->x[i][a] =
                            coordinates[tag][a] + (side ? -1 : 1) * 1.0e-6 * coordinates[tag][b];
                    }
                    commands(lmp, "run 0");
                    values[side] = energy(lmp);
                }
                for (int i = 0; i < lmp->atom->nlocal; ++i)
                    lmp->atom->x[i][a] = coordinates[lmp->atom->tag[i] - 1][a];
                commands(lmp, "run 0");
                EXPECT_NEAR(stress[tensor[a][b]], -(values[0] - values[1]) / 2.0e-6, 1.0e-8);
            }
    }
    lammps_close(lmp);
}
