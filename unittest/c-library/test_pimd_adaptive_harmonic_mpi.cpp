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
#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

using namespace LAMMPS_NS;
namespace {
constexpr int atoms = 32, beads = 4, observables = 5;
// Independent one-body oracle, E=sum_i |q_i|^2/2. No adaptive helper is used.
class Trap : public Pair {
public:
    explicit Trap(LAMMPS *lmp) : Pair(lmp)
    {
        single_enable = restartinfo = 0;
        no_virial_fdotr_compute     = 1;
    }
    ~Trap() override
    {
        if (allocated) {
            memory->destroy(setflag);
            memory->destroy(cutsq);
        }
    }
    void settings(int narg, char **) override
    {
        if (narg) error->all(FLERR, "Trap takes no arguments");
    }
    void coeff(int narg, char **) override
    {
        if (narg != 2) error->all(FLERR, "Trap requires pair_coeff * *");
        if (!allocated) {
            allocated = 1;
            memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "adaptive_trap:setflag");
            memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "adaptive_trap:cutsq");
        }
        for (int i = 1; i <= atom->ntypes; ++i)
            for (int j = 1; j <= atom->ntypes; ++j)
                setflag[i][j] = 1;
    }
    void init_style() override {}
    double init_one(int, int) override { return 0.0; }
    void compute(int eflag, int vflag) override
    {
        ev_init(eflag, vflag);
        for (int i = 0; i < atom->nlocal; ++i) {
            double q[3];
            domain->unmap(atom->x[i], atom->image[i], q);
            double energy = 0.0;
            for (int d = 0; d < 3; ++d) {
                atom->f[i][d] -= q[d];
                energy += 0.5 * q[d] * q[d];
            }
            if (eflag_global) eng_vdwl += energy;
            if (eflag_atom) eatom[i] += energy;
        }
    }
};
Pair *create_trap(LAMMPS *lmp)
{
    return new Trap(lmp);
}
struct Registration {
    Registration() { Force::pair_styles().set_plugin("adaptive/trap/test", create_trap); }
    ~Registration() { Force::pair_styles().clear_plugin("adaptive/trap/test"); }
};

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
std::array<double, observables> reference(int geometry)
{
    // Integrate the radial centroid marginal analytically marginalized over
    // all 3(P-1) Gaussian modes. Simpson quadrature on [0,12].
    std::array<double, observables> result{};
    constexpr int intervals = 24000;
    const double h = 12.0 / intervals, pi = std::acos(-1.0);
    double norm = 0.0;
    for (int j = 0; j <= intervals; ++j) {
        const double r = j * h;
        double a       = 1.0;
        if (r >= 2.5)
            a = 0.0;
        else if (r > 0.5) {
            const double t = (r - 0.5) / 2.0;
            a              = 1.0 - 10 * t * t * t + 15 * t * t * t * t - 6 * t * t * t * t * t;
        }
        if (geometry == 2) a = 1.0;
        if (geometry == 3) a = 0.0;
        double correction = 0.0, internal = 0.0;
        for (int k = 1; k < beads; ++k) {
            const double eigenvalue = 4 * std::pow(std::sin(pi * k / beads), 2);
            const double spring     = beads * eigenvalue; // m=T=hbar=1
            correction += 1.5 * std::log1p(a * a / (beads * spring));
            internal += 3.0 / (beads * (spring + a * a / beads));
        }
        const double jacobian = geometry == 1 ? 1.0 : r * r;
        const double w        = (j == 0 || j == intervals ? 1.0 : (j % 2 ? 4.0 : 2.0)) * jacobian *
                                std::exp(-0.5 * r * r - correction);
        const double r2       = r * r + (geometry == 1 ? 2.0 : 0.0);
        const double r4       = r * r * r * r + (geometry == 1 ? 4.0 * r * r + 8.0 : 0.0);
        const double values[observables] = {r2, r4, r2 + a * a * internal, internal, a};
        norm += w;
        for (int o = 0; o < observables; ++o)
            result[o] += w * values[o];
    }
    for (auto &v : result)
        v /= norm;
    return result;
}
} // namespace

void run_statistics(int geometry)
{
    Registration registration;
    const auto exact   = reference(geometry);
    const char *output = std::getenv("LAMMPS_ADAPTIVE_STATISTICS_FILE");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    SafeFilePtr file;
    const std::string filename =
        output ? std::string(output) + (geometry ? "." + std::to_string(geometry) : "") : "";
    if (output && rank == 0) file = std::fopen(filename.c_str(), "w");
    int io_error = (output && rank == 0 && !file);
    MPI_Bcast(&io_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ASSERT_EQ(io_error, 0);
    if (file) utils::print(file, "dt,replica,r2,r4,q2,internal2,amplitude\n");
    for (int level = 0; level < 2; ++level) {
        const double dt    = level ? 0.02 : 0.04;
        const char *args[] = {"LAMMPS_test", "-screen", "none", "-log", "none",
                              "-partition",  "4x1",     "-in",  "none", "-nocite"};
        auto *lmp          = static_cast<LAMMPS *>(
            lammps_open(10, const_cast<char **>(args), MPI_COMM_WORLD, nullptr));
        std::string input = "units lj\natom_style atomic\nboundary f f f\n"
                            "atom_modify map array sort 0 0\nregion box block -12 12 -12 12 -12 "
                            "12\ncreate_box 1 box\n";
        for (int i = 0; i < atoms; ++i)
            input += "create_atoms 1 single " + std::to_string(0.01 * i) + " 0.1 -0.2\n";
        input += "mass 1 1\npair_style adaptive/trap/test\npair_coeff * *\n"
                 "neighbor 0.3 nsq\nneigh_modify every 1 delay 0 check no\n"
                 "velocity all set 0 0 0\ntimestep " +
                 std::to_string(dt) +
                 "\n"
                 "fix pi all pimd/adaptive temp 1 sp 6.283185307179586 " +
                 std::string(geometry == 3 ? "center 100 0 0 " : "center 0 0 0 ") +
                 std::string(geometry == 1 ? "geometry slab z " : "") +
                 std::string(geometry == 2 ? "core 20 outer 30 " : "core 0.5 outer 2.5 ") +
                 "initialize auxiliary thermostat pile 1 " + std::to_string(23117 + 1000 * level) +
                 "\n"
                 "thermo 100000000\nrun 0\nrun 5000 pre no post no";
        bool ok = commands(lmp, input);
        std::array<double, atoms * observables> totals{};
        int samples = 0;
        if (ok) {
            auto *fix = lmp->modify->get_fix_by_id("pi");
            for (int block = 0; block < 5000; ++block) {
                if (!commands(lmp, "run 10 pre no post no")) {
                    ok = false;
                    break;
                }
                std::array<double, atoms * 6> current{};
                for (int i = 0; i < lmp->atom->nlocal; ++i) {
                    const int base = 6 * (lmp->atom->tag[i] - 1);
                    double q[3];
                    lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], q);
                    for (int d = 0; d < 3; ++d) {
                        current[base + d] += fix->array_atom[i][d] / beads;
                        current[base + 3] += q[d] * q[d] / beads;
                        current[base + 4] += fix->array_atom[i][d] * fix->array_atom[i][d] / beads;
                    }
                    current[base + 5] += fix->array_atom[i][9] / beads;
                }
                MPI_Allreduce(MPI_IN_PLACE, current.data(), current.size(), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                for (int i = 0; i < atoms; ++i) {
                    const int base = 6 * i;
                    double r2      = 0.0;
                    for (int d = 0; d < 3; ++d)
                        r2 += current[base + d] * current[base + d];
                    const double values[observables] = {r2, r2 * r2, current[base + 3],
                                                        current[base + 4] - r2, current[base + 5]};
                    for (int o = 0; o < observables; ++o)
                        totals[observables * i + o] += values[o];
                }
                ++samples;
            }
        }
        lammps_close(lmp);
        ASSERT_TRUE(ok);
        ASSERT_EQ(samples, 5000);
        for (auto &v : totals)
            v /= samples;
        for (int i = 0; i < atoms; ++i)
            if (file)
                utils::print(file, "{},{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}\n", dt, i,
                             totals[i * observables], totals[i * observables + 1],
                             totals[i * observables + 2], totals[i * observables + 3],
                             totals[i * observables + 4]);
        for (int o = 0; o < observables; ++o) {
            double mean = 0.0, variance = 0.0;
            for (int i = 0; i < atoms; ++i)
                mean += totals[i * observables + o] / atoms;
            for (int i = 0; i < atoms; ++i)
                variance += std::pow(totals[i * observables + o] - mean, 2) / (atoms - 1);
            const double limit =
                6.0 * std::sqrt(variance / atoms) + 0.005 * std::max(0.1, std::abs(exact[o]));
            EXPECT_NEAR(mean, exact[o], limit) << "dt=" << dt << " observable=" << o;
            if (rank == 0)
                utils::print(
                    stdout,
                    "ADAPTIVE_STAT geometry={} dt={} observable={} mean={:.10g} reference={:.10g} "
                    "limit={:.6g}\n",
                    geometry, dt, o, mean, exact[o], limit);
        }
    }
}

TEST(PIMDAdaptiveStatistics, spherical_harmonic_two_timesteps)
{
    run_statistics(0);
}

TEST(PIMDAdaptiveStatistics, slab_harmonic_two_timesteps)
{
    run_statistics(1);
}

TEST(PIMDAdaptiveStatistics, uniform_quantum_harmonic_two_timesteps)
{
    run_statistics(2);
}

TEST(PIMDAdaptiveStatistics, classical_harmonic_two_timesteps)
{
    run_statistics(3);
}

namespace {
// Two coupled trapped particles. The second particle's quantum amplitude follows
// its distance to the moving first particle. No production map helper is used
// by the independent Gaussian marginal below.
constexpr double relative_k = 0.7, center_k = 0.4;
constexpr int coupled_observables = 7, independent_replicas = 8;

class CoupledTrap : public Trap {
public:
    explicit CoupledTrap(LAMMPS *lmp) : Trap(lmp) {}
    void compute(int eflag, int vflag) override
    {
        ev_init(eflag, vflag);
        if (atom->nlocal != 2)
            error->all(FLERR, "Coupled trap test requires two locally owned atoms");
        int index[2] = {-1, -1};
        for (int i = 0; i < atom->nlocal; ++i)
            if ((atom->tag[i] >= 1) && (atom->tag[i] <= 2)) index[atom->tag[i] - 1] = i;
        if ((index[0] < 0) || (index[1] < 0))
            error->all(FLERR, "Coupled trap test requires atom IDs one and two");
        double x[2][3];
        for (int i = 0; i < 2; ++i)
            domain->unmap(atom->x[index[i]], atom->image[index[i]], x[i]);
        double energy = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double delta = x[1][d] - x[0][d], sum = x[1][d] + x[0][d];
            energy += 0.5 * (relative_k * delta * delta + center_k * sum * sum);
            atom->f[index[0]][d] += relative_k * delta - center_k * sum;
            atom->f[index[1]][d] -= relative_k * delta + center_k * sum;
        }
        if (eflag_global) eng_vdwl = energy;
        if (eflag_atom)
            for (int i = 0; i < 2; ++i)
                eatom[index[i]] = 0.5 * energy;
    }
};

Pair *create_coupled_trap(LAMMPS *lmp)
{
    return new CoupledTrap(lmp);
}

struct CoupledRegistration {
    CoupledRegistration()
    {
        Force::pair_styles().set_plugin("adaptive/coupled/test", create_coupled_trap);
    }
    ~CoupledRegistration() { Force::pair_styles().clear_plugin("adaptive/coupled/test"); }
};

std::array<double, coupled_observables> coupled_reference(int intervals)
{
    std::array<double, coupled_observables> result{};
    const double step = 12.0 / intervals, pi = std::acos(-1.0);
    double norm = 0.0;
    for (int j = 0; j <= intervals; ++j) {
        const double r = step * j;
        double a       = 1.0;
        if (r >= 2.5)
            a = 0.0;
        else if (r > 0.5) {
            const double t = (r - 0.5) / 2.0;
            a              = 1.0 - 10 * t * t * t + 15 * std::pow(t, 4) - 6 * std::pow(t, 5);
        }
        double logdet = 0.0, spread = 0.0;
        for (int k = 1; k < beads; ++k) {
            const double spring = 4.0 * beads * std::pow(std::sin(pi * k / beads), 2);
            const double h00    = spring + (relative_k + center_k) / beads;
            const double h11    = spring + a * a * (relative_k + center_k) / beads;
            const double h01    = a * (center_k - relative_k) / beads;
            const double det    = h00 * h11 - h01 * h01;
            logdet += 1.5 * std::log(det / (h00 * spring));
            spread += 3.0 * a * a * h00 / (beads * det);
        }
        const double w = (j == 0 || j == intervals ? 1 : (j % 2 ? 4 : 2)) * r * r *
                         std::exp(-0.5 * relative_k * r * r - logdet);
        const double values[coupled_observables] = {
            r * r, std::pow(r, 4),      3.0 / (4.0 * center_k), spread,
            a,     r < 0.5 ? 1.0 : 0.0, r > 2.5 ? 1.0 : 0.0};
        norm += w;
        for (int o = 0; o < coupled_observables; ++o)
            result[o] += w * values[o];
    }
    for (auto &v : result)
        v /= norm;
    return result;
}
} // namespace

TEST(PIMDAdaptiveStatistics, coupled_harmonic_moving_reference_two_timesteps)
{
    CoupledRegistration registration;
    const auto exact = coupled_reference(48000), coarse = coupled_reference(24000);
    for (int o = 0; o < coupled_observables; ++o)
        EXPECT_NEAR(coarse[o], exact[o], o >= 5 ? 5e-4 : 1e-9);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int level = 0; level < 2; ++level) {
        const double dt = level ? 0.02 : 0.04;
        std::array<std::array<double, coupled_observables>, independent_replicas> totals{};
        for (int replica = 0; replica < independent_replicas; ++replica) {
            const char *args[] = {"LAMMPS_test", "-screen", "none", "-log", "none",
                                  "-partition",  "4x1",     "-in",  "none", "-nocite"};
            auto *lmp          = static_cast<LAMMPS *>(
                lammps_open(10, const_cast<char **>(args), MPI_COMM_WORLD, nullptr));
            const std::string input =
                "units lj\natom_style atomic\nboundary f f f\n"
                "atom_modify map array sort 0 0\nregion box block -12 12 -12 12 -12 12\n"
                "create_box 1 box\ncreate_atoms 1 single -0.5 0.2 0\n"
                "create_atoms 1 single 0.5 -0.2 0\nmass 1 1\n"
                "group anchor id 1\npair_style adaptive/coupled/test\npair_coeff * *\n"
                "neighbor 0.3 nsq\nneigh_modify every 1 delay 0 check no\n"
                "velocity all set 0 0 0\ntimestep " +
                std::to_string(dt) +
                "\nfix pi all pimd/adaptive temp 1 sp 6.283185307179586 "
                "center group anchor core 0.5 outer 2.5 initialize auxiliary "
                "thermostat pile 1 " +
                std::to_string(71237 + 1000 * level + 100 * replica) +
                "\nthermo 100000000\nrun 0\nrun 10000 pre no post no";
            bool ok     = commands(lmp, input);
            int samples = 0, transitions = 0, last_region = -1;
            if (ok) {
                auto *fix = lmp->modify->get_fix_by_id("pi");
                for (int block = 0; block < 10000; ++block) {
                    if (!commands(lmp, "run 10 pre no post no")) {
                        ok = false;
                        break;
                    }
                    std::array<double, 8> state{};
                    for (int i = 0; i < lmp->atom->nlocal; ++i) {
                        const int tag = lmp->atom->tag[i] - 1;
                        for (int d = 0; d < 3; ++d)
                            state[3 * tag + d] += fix->array_atom[i][d] / beads;
                        if (tag == 1) {
                            double x[3];
                            lmp->domain->unmap(lmp->atom->x[i], lmp->atom->image[i], x);
                            for (double v : x)
                                state[6] += v * v / beads;
                            state[7] += fix->array_atom[i][9] / beads;
                        }
                    }
                    MPI_Allreduce(MPI_IN_PLACE, state.data(), state.size(), MPI_DOUBLE, MPI_SUM,
                                  MPI_COMM_WORLD);
                    double r2 = 0.0, com2 = 0.0, c12 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        r2 += std::pow(state[3 + d] - state[d], 2);
                        com2 += 0.25 * std::pow(state[3 + d] + state[d], 2);
                        c12 += state[3 + d] * state[3 + d];
                    }
                    const int region = r2 < 0.25 ? 0 : (r2 > 6.25 ? 2 : 1);
                    if ((last_region >= 0) && (region != last_region)) ++transitions;
                    last_region                              = region;
                    const double values[coupled_observables] = {r2,
                                                                r2 * r2,
                                                                com2,
                                                                state[6] - c12,
                                                                state[7],
                                                                region == 0 ? 1.0 : 0.0,
                                                                region == 2 ? 1.0 : 0.0};
                    for (int o = 0; o < coupled_observables; ++o)
                        totals[replica][o] += values[o];
                    ++samples;
                }
            }
            lammps_close(lmp);
            ASSERT_TRUE(ok);
            ASSERT_EQ(samples, 10000);
            EXPECT_GT(transitions, 20);
            for (auto &v : totals[replica])
                v /= samples;
        }
        for (int o = 0; o < coupled_observables; ++o) {
            double mean = 0.0, variance = 0.0;
            for (const auto &sample : totals)
                mean += sample[o] / independent_replicas;
            for (const auto &sample : totals)
                variance += std::pow(sample[o] - mean, 2) / (independent_replicas - 1);
            const double limit = 6.0 * std::sqrt(variance / independent_replicas) +
                                 0.005 * std::max(0.1, std::abs(exact[o]));
            EXPECT_NEAR(mean, exact[o], limit) << "dt=" << dt << " observable=" << o;
            if (rank == 0)
                utils::print(stdout,
                             "COUPLED_ADAPTIVE_STAT dt={} observable={} mean={:.10g} "
                             "reference={:.10g} limit={:.6g}\n",
                             dt, o, mean, exact[o], limit);
        }
    }
}
