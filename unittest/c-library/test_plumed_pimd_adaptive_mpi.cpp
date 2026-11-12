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

// Actual PLUMED gradients and history for the explicit adaptive ensemble.
#define LAMMPS_LIB_MPI 1
#include "atom.h"
#include "fix.h"
#include "lammps.h"
#include "library.h"
#include "modify.h"
#include "universe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"
using namespace LAMMPS_NS;
namespace {
using Positions = std::array<double, 36>;
Positions coordinates(int coarse)
{
    Positions q{};
    for (int b = 0; b < 4; ++b) {
        q[9 * b]     = 5.7 + 0.2 * b;
        q[9 * b + 1] = 4.1 + 0.03 * b;
        q[9 * b + 2] = 4.3;
        q[9 * b + 3] = 4.8 + 0.3 * (b / (4 / coarse));
        q[9 * b + 4] = 4.4;
        q[9 * b + 5] = 4.5;
        q[9 * b + 6] = 4.0;
        q[9 * b + 7] = 4.7;
        q[9 * b + 8] = 4.2;
    }
    return q;
}
double oracle(Positions q, const std::string &mode)
{
    Positions center = q;
    for (int i = 0; i < 9; ++i) {
        double mean = 0;
        for (int b = 0; b < 4; ++b)
            mean += q[9 * b + i] / 4;
        for (int b = 0; b < 4; ++b)
            center[9 * b + i] = mean;
    }
    double sc = 0;
    for (int i = 0; i < 6; ++i)
        sc += std::pow(center[i] - center[i + 3], 2);
    if (mode == "coordinate_mean") return 0.01 * sc * sc;
    double mean = 0, energy = 0, density = 0;
    for (int b = 0; b < 4; ++b) {
        double s = 0;
        for (int i = 0; i < 6; ++i)
            s += std::pow(q[9 * b + i] - q[9 * b + i + 3], 2);
        mean += s / 4;
        energy += 0.01 * s * s / 4;
        density += std::exp(-0.01 * s * s) / 4;
    }
    if (mode == "cv_mean") return 0.01 * mean * mean;
    if (mode == "bias_mean") return energy;
    if (mode == "probability") return -std::log(density);
    return -std::log(0.5 * std::exp(-0.01 * sc * sc) + 0.5 * density);
}
// Independent scalar definition for two competing centers and one assigned atom.
double localActivity(const double *q, bool outside = false, bool slab = false)
{
    double d12 = 0, d32 = 0, radius = 0;
    for (int d = 0; d < 3; ++d) {
        d12 += std::pow(q[d] - q[3 + d], 2);
        d32 += std::pow(q[6 + d] - q[3 + d], 2);
        if (!slab || d == 0) radius += std::pow(q[d] - q[6 + d], 2);
    }
    const double first  = std::exp(-1.2 * std::sqrt(d12));
    const double other  = std::exp(-1.2 * std::sqrt(d32));
    const double defect = first / (first + other) - 0.5;
    const double t      = std::max(0.0, std::min(1.0, (std::sqrt(radius) - 1.5) / 1.3));
    const double chi    = 1 - 10 * std::pow(t, 3) + 15 * std::pow(t, 4) - 6 * std::pow(t, 5);
    return (outside ? 1 - chi : 1 + chi) * defect * defect;
}

double localOracle(const Positions &q, const std::string &mode)
{
    if (mode == "coordinate_mean") {
        std::array<double, 9> centroid{};
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 9; ++i)
                centroid[i] += q[9 * b + i] / 4;
        const double s = localActivity(centroid.data());
        return 0.01 * s * s;
    }
    double average = 0, energy = 0;
    for (int b = 0; b < 4; ++b) {
        const double s = localActivity(q.data() + 9 * b);
        average += s / 4;
        energy += 0.01 * s * s / 4;
    }
    return mode == "cv_mean" ? 0.01 * average * average : energy;
}

// Independent mapping used only to evaluate the scalar bias oracle.
Positions physical(Positions y, bool mobile = false, bool slab = false)
{
    std::array<double, 3> anchor{4, 4, 4};
    if (mobile) {
        anchor = {};
        for (int b = 0; b < 4; ++b)
            for (int d = 0; d < 3; ++d)
                anchor[d] += (2 * y[9 * b + 3 + d] + 3 * y[9 * b + 6 + d]) / 20;
    }
    for (int i = 0; i < 3; ++i) {
        double c[3] = {};
        for (int b = 0; b < 4; ++b)
            for (int d = 0; d < 3; ++d)
                c[d] += y[9 * b + 3 * i + d] / 4;
        double r2 = 0;
        for (int d = 0; d < 3; ++d)
            if (!slab || d == 0) r2 += (c[d] - anchor[d]) * (c[d] - anchor[d]);
        const double t = std::max(0.0, std::min(1.0, (std::sqrt(r2) - 0.2) / 2.6));
        const double a = 1 - 10 * std::pow(t, 3) + 15 * std::pow(t, 4) - 6 * std::pow(t, 5);
        for (int b = 0; b < 4; ++b)
            for (int d = 0; d < 3; ++d)
                y[9 * b + 3 * i + d] = c[d] + a * (y[9 * b + 3 * i + d] - c[d]);
    }
    return y;
}

Positions contracted(Positions q, double lambda)
{
    for (int i = 0; i < 9; ++i) {
        double centroid = 0;
        for (int b = 0; b < 4; ++b)
            centroid += q[9 * b + i] / 4;
        for (int b = 0; b < 4; ++b)
            q[9 * b + i] = centroid + lambda * (q[9 * b + i] - centroid);
    }
    return q;
}

class PlumedAdaptiveMPI : public ::testing::Test {
protected:
    bool localCV = false;
    bool mobile = false, slab = false;
    double contraction = -1;
    bool outsideWall = false, localSlab = false;
    LAMMPS *lmp        = nullptr;
    int rank           = 0;
    std::string prefix = "adaptive_plumed";
    void SetUp() override
    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        ASSERT_EQ(size, 8);
    }
    void close()
    {
        if (lmp) {
            lammps_close(lmp);
            lmp = nullptr;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    void TearDown() override { close(); }
    bool command(const std::string &text)
    {
        lammps_commands_string(lmp, text.c_str());
        int bad = lammps_has_error(lmp);
        char msg[1024]{};
        if (bad) lammps_get_last_error_message(lmp, msg, sizeof(msg));
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        EXPECT_EQ(bad, 0) << msg;
        return !bad;
    }
    void open()
    {
        LAMMPS::argv args = {"adaptive_plumed", "-screen", "none", "-log", "none",
                             "-partition",      "4x2",     "-in",  "none", "-nocite"};
        lmp               = new LAMMPS(args, MPI_COMM_WORLD);
        lammps_set_show_error(lmp, 0);
    }
    bool prepare(int coarse, bool restart = false)
    {
        open();
        if (restart) {
            if (!command(
                    "processors 2 1 1\nvariable file world adaptive_plumed.0 adaptive_plumed.1 "
                    "adaptive_plumed.2 adaptive_plumed.3\nread_restart ${file}\natom_modify sort 0 "
                    "0"))
                return false;
        } else {
            const auto q = coordinates(coarse);
            const int b  = rank / 2;
            std::ostringstream in;
            in << std::setprecision(17)
               << "units lj\natom_style atomic\natom_modify map array sort 0 0\nprocessors 2 1 1\n"
               << "region box block 0 12 0 12 0 12\ncreate_box 3 box\n";
            for (int i = 0; i < 3; ++i)
                in << "create_atoms " << i + 1 << " single " << q[9 * b + 3 * i] << " "
                   << q[9 * b + 3 * i + 1] << " " << q[9 * b + 3 * i + 2] << "\n";
            in << "mass 1 1\nmass 2 2\nmass 3 3\npair_style zero 2.5\npair_coeff * *\n"
               << "group fine id 1\ngroup coarse id 2\nvelocity all set 0 0 0\n";
            if (!command(in.str())) return false;
        }
        if (mobile && !command("group site id 2 3")) return false;
        return command("timestep 0.0001\nthermo 100000\nfix pi all pimd/adaptive temp 1 sp "
                       "6.283185307179586 center " +
                       std::string(mobile ? "group site" : "4 4 4") +
                       " core 0.2 outer 2.8 initialize auxiliary" +
                       (slab ? " geometry slab x" : ""));
    }
    void input(const std::string &mode, bool metad = false, bool restart = false)
    {
        if (rank == 0) {
            std::ofstream out(prefix + ".dat");
            if (restart) out << "RESTART\n";
            if (localCV)
                out << "s: VORONOI_LOCAL_COORDINATION CENTERS=1,3 ASSIGNED=2 "
                       "REGION_CENTER=3 REFERENCE=0.5 KAPPA=1.2 CORE_RADIUS=1.5 OUTER_RADIUS=2.8 "
                       "PAIR_SWITCH=2.5 PAIR_CUTOFF=3.5 SKIN=0.2 NL_STRIDE=10 NOPBC"
                    << (outsideWall ? " REGION_SIDE=OUTSIDE" : "")
                    << (localSlab ? " REGION_GEOMETRY=SLAB_X" : "") << "\n";
            else
                out << "d1: DISTANCE ATOMS=1,2 NOPBC\nd2: DISTANCE ATOMS=2,3 NOPBC\n"
                    << "s: CUSTOM ARG=d1,d2 FUNC=x*x+y*y PERIODIC=NO\n";
            if (mode == "cv_mean") out << "m: ENSEMBLE ARG=s\n";
            if (mode == "probability" || mode == "mixture") {
                out << "v: CUSTOM ARG=s FUNC=0.01*x*x PERIODIC=NO\n"
                    << "rho: CUSTOM ARG=v FUNC=exp(-x) PERIODIC=NO\nr: ENSEMBLE ARG=rho\n";
                if (mode == "probability")
                    out << "e: CUSTOM ARG=r.rho FUNC=-log(x) PERIODIC=NO\n";
                else
                    out << "dx: DISTANCE ATOMS=1,2 COMPONENTS NOPBC\ndy: DISTANCE ATOMS=2,3 "
                           "COMPONENTS NOPBC\n"
                        << "cm: ENSEMBLE ARG=dx.x,dx.y,dx.z,dy.x,dy.y,dy.z\n"
                        << "sc: CUSTOM ARG=cm.dx.x,cm.dx.y,cm.dx.z,cm.dy.x,cm.dy.y,cm.dy.z "
                           "VAR=u1,u2,u3,u4,u5,u6 FUNC=u1*u1+u2*u2+u3*u3+u4*u4+u5*u5+u6*u6 "
                           "PERIODIC=NO\n"
                        << "bc: CUSTOM ARG=sc FUNC=0.01*x*x PERIODIC=NO\n"
                        << "e: CUSTOM ARG=bc,r.rho FUNC=-log(0.5*exp(-x)+0.5*y) PERIODIC=NO\n";
                out << "b: BIASVALUE ARG=e\n";
            } else if (outsideWall) {
                out << "b: UPPER_WALLS ARG=s AT=0 KAPPA=2 EXP=2 EPS=1\n";
            } else if (metad) {
                out << "b: METAD ARG=" << (mode == "cv_mean" ? "m.s" : "s")
                    << " SIGMA=0.5 HEIGHT=" << (mode == "bias_mean" ? 0.0625 : 0.25)
                    << " PACE=2 FILE=" << prefix << ".hills RESTART=" << (restart ? "YES" : "NO");
                if (mode == "bias_mean") out << " WALKERS_MPI";
                out << "\nFLUSH STRIDE=1\n";
            } else
                out << "b: RESTRAINT ARG=" << (mode == "cv_mean" ? "m.s" : "s")
                    << " AT=0 KAPPA=0.02\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    bool attach(const std::string &mode)
    {
        return command("fix pl all plumed plumedfile " + prefix + ".dat outfile " + prefix +
                       ".log path_integral " +
                       ((mode == "probability" || mode == "mixture") ? "cv_mean" : mode) +
                       " pimd_fix pi" +
                       std::string(outsideWall ? " path_coordinates physical" : "") +
                       (contraction < 0 ? "" : " path_contraction " + std::to_string(contraction)));
    }
    double bias() { return lmp->modify->get_fix_by_id("pi")->compute_vector(4); }
    std::array<double, 108> state()
    {
        std::array<double, 108> a{};
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const int j = 27 * (rank / 2) + 9 * (lmp->atom->tag[i] - 1);
            for (int d = 0; d < 3; ++d) {
                a[j + d]     = lmp->atom->x[i][d];
                a[j + 3 + d] = lmp->atom->v[i][d];
                a[j + 6 + d] = lmp->modify->get_fix_by_id("pi")->array_atom[i][6 + d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, a.data(), a.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return a;
    }
    bool place(const Positions &y)
    {
        const auto q = physical(y, mobile, slab);
        auto *pi     = lmp->modify->get_fix_by_id("pi");
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            for (int d = 0; d < 3; ++d) {
                const int j          = 9 * (rank / 2) + 3 * (lmp->atom->tag[i] - 1) + d;
                pi->array_atom[i][d] = y[j];
                lmp->atom->x[i][d]   = q[j];
            }
        return command("run 0");
    }
    std::vector<std::vector<double>> hills(const std::string &suffix)
    {
        std::ifstream file(prefix + ".hills" + suffix);
        EXPECT_TRUE(file.good());
        std::vector<std::vector<double>> data;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            std::vector<double> vals;
            double x;
            while (row >> x)
                vals.push_back(x);
            if (!vals.empty()) data.push_back(vals);
        }
        return data;
    }
    void clean()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0 && !HasFailure()) {
            std::remove((prefix + ".dat").c_str());
            for (const auto &ext : {".log", ".hills"})
                for (const auto &suffix : {"", ".0", ".1", ".2", ".3"})
                    std::remove((prefix + ext + suffix).c_str());
            for (int b = 0; b < 4; ++b)
                std::remove(("adaptive_plumed." + std::to_string(b)).c_str());
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

TEST_F(PlumedAdaptiveMPI, nonlinear_local_and_complete_path_gradients)
{
    for (bool local : {false, true}) {
        localCV = local;
        for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
            SCOPED_TRACE(mode + (local ? ":local" : ":distance"));
            prefix = "adaptive_gradient_" + mode + (local ? "_local" : "_distance");
            input(mode);
            ASSERT_TRUE(prepare(2));
            ASSERT_TRUE(attach(mode));
            ASSERT_TRUE(command("run 0"));
            const auto y      = coordinates(2);
            const auto force  = state();
            const auto energy = [&](const Positions &z) {
                return local ? localOracle(physical(z), mode) : oracle(physical(z), mode);
            };
            EXPECT_NEAR(bias(), energy(y), 1.0e-11);
            auto *pi = lmp->modify->get_fix_by_id("pi");
            EXPECT_NEAR(pi->compute_vector(0), 0, 1.0e-11);
            for (int j = 0; j < 36; ++j) {
                auto plus = y, minus = y;
                plus[j] += 1.0e-5;
                minus[j] -= 1.0e-5;
                const double expected = -4 * (energy(plus) - energy(minus)) / 2.0e-5;
                const int b = j / 9, i = (j % 9) / 3, d = j % 3;
                EXPECT_NEAR(force[27 * b + 9 * i + 6 + d], expected, 5.0e-8);
                ASSERT_TRUE(place(plus));
                const double ep = bias();
                ASSERT_TRUE(place(minus));
                const double em = bias();
                EXPECT_NEAR(-4 * (ep - em) / 2.0e-5, expected, 5.0e-8);
            }
            ASSERT_TRUE(place(y));
            std::array<double, 6> virial{};
            auto *pl = lmp->modify->get_fix_by_id("pl");
            for (int k = 0; k < 6; ++k)
                virial[k] = pl->virial[k] / 4;
            MPI_Allreduce(MPI_IN_PLACE, virial.data(), 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            const int row[6] = {0, 1, 2, 0, 0, 1}, col[6] = {0, 1, 2, 1, 2, 2};
            for (int k = 0; k < 6; ++k) {
                auto plus = y, minus = y;
                for (int b = 0; b < 4; ++b)
                    for (int i = 0; i < 3; ++i) {
                        plus[9 * b + 3 * i + col[k]] += 1.0e-5 * y[9 * b + 3 * i + row[k]];
                        minus[9 * b + 3 * i + col[k]] -= 1.0e-5 * y[9 * b + 3 * i + row[k]];
                    }
                EXPECT_NEAR(virial[k], -(energy(plus) - energy(minus)) / 2.0e-5, 1.0e-7);
            }
            ASSERT_TRUE(command("fix_modify pl energy no\nrun 0"));
            EXPECT_NEAR(pi->compute_vector(0), 0, 1.0e-11);
            EXPECT_NEAR(pi->compute_scalar(),
                        pi->compute_vector(1) + pi->compute_vector(2) + energy(y), 1.0e-11);
            close();
            clean();
        }
    }
}
TEST_F(PlumedAdaptiveMPI, normalized_metad_history_repeated_step_and_restart)
{
    for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
        SCOPED_TRACE(mode);
        prefix = "adaptive_plumed_history_" + mode;
        clean();
        input(mode, true);
        ASSERT_TRUE(prepare(2));
        ASSERT_TRUE(attach(mode));
        ASSERT_TRUE(command("run 2\nrun 0"));
        const auto before    = state();
        const double initial = bias();
        if (rank == 0) {
            const int copies = mode == "cv_mean" ? 4 : 1;
            for (int c = 0; c < copies; ++c) {
                const auto rows = hills(mode == "cv_mean" ? "." + std::to_string(c) : "");
                EXPECT_EQ(rows.size(), mode == "bias_mean" ? 4 : 1);
                double height = 0;
                for (const auto &row : rows) {
                    EXPECT_GE(row.size(), 4);
                    if (row.size() >= 4) height += row[3];
                }
                EXPECT_NEAR(height, 0.25, 1e-12);
            }
        }
        ASSERT_TRUE(command("run 0"));
        const auto repeated = state();
        for (int i = 0; i < 108; ++i)
            EXPECT_NEAR(repeated[i], before[i], 1e-12);
        ASSERT_TRUE(
            command("variable file world adaptive_plumed.0 adaptive_plumed.1 adaptive_plumed.2 "
                    "adaptive_plumed.3\nwrite_restart ${file}"));
        close();
        input(mode, true, true);
        ASSERT_TRUE(prepare(2, true));
        ASSERT_TRUE(attach(mode));
        ASSERT_TRUE(command("run 0"));
        const auto restored = state();
        EXPECT_NEAR(bias(), initial, 1e-10);
        for (int i = 0; i < 108; ++i)
            EXPECT_NEAR(restored[i], before[i], 1e-10);
        ASSERT_TRUE(command("run 2"));
        close();
        if (rank == 0) {
            for (int c = 0; c < (mode == "cv_mean" ? 4 : 1); ++c) {
                const auto rows = hills(mode == "cv_mean" ? "." + std::to_string(c) : "");
                EXPECT_EQ(rows.size(), mode == "bias_mean" ? 8 : 2);
            }
        }
        clean();
    }
}
TEST_F(PlumedAdaptiveMPI, local_voronoi_shared_history_and_restart)
{
    localCV = true;
    for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
        SCOPED_TRACE(mode);
        prefix = "adaptive_local_history_" + mode;
        clean();
        input(mode, true);
        ASSERT_TRUE(prepare(2));
        ASSERT_TRUE(attach(mode));
        ASSERT_TRUE(command("run 2\nrun 0"));
        const auto before    = state();
        const double initial = bias();
        if (rank == 0) {
            const int copies = mode == "cv_mean" ? 4 : 1;
            for (int c = 0; c < copies; ++c) {
                const auto rows = hills(mode == "cv_mean" ? "." + std::to_string(c) : "");
                EXPECT_EQ(rows.size(), mode == "bias_mean" ? 4 : 1);
                double height = 0;
                for (const auto &row : rows) {
                    EXPECT_GE(row.size(), 4);
                    if (row.size() >= 4) height += row[3];
                }
                EXPECT_NEAR(height, 0.25, 1e-12);
            }
        }
        ASSERT_TRUE(command("run 0"));
        const auto repeated = state();
        for (int i = 0; i < 108; ++i)
            EXPECT_NEAR(repeated[i], before[i], 1e-12);
        ASSERT_TRUE(
            command("variable file world adaptive_plumed.0 adaptive_plumed.1 adaptive_plumed.2 "
                    "adaptive_plumed.3\nwrite_restart ${file}"));
        close();
        input(mode, true, true);
        ASSERT_TRUE(prepare(2, true));
        ASSERT_TRUE(attach(mode));
        ASSERT_TRUE(command("run 0"));
        const auto restored = state();
        EXPECT_NEAR(bias(), initial, 1e-10);
        for (int i = 0; i < 108; ++i)
            EXPECT_NEAR(restored[i], before[i], 1e-10);
        ASSERT_TRUE(command("run 2"));
        close();
        if (rank == 0) {
            for (int c = 0; c < (mode == "cv_mean" ? 4 : 1); ++c) {
                const auto rows = hills(mode == "cv_mean" ? "." + std::to_string(c) : "");
                EXPECT_EQ(rows.size(), mode == "bias_mean" ? 8 : 2);
            }
        }
        clean();
    }
}
} // namespace

TEST_F(PlumedAdaptiveMPI, mobile_slab_contracted_gradients_and_strain)
{
    for (bool moving : {false, true}) {
        mobile = moving;
        slab   = moving;
        for (bool local : {false, true}) {
            localCV = local;
            for (double lambda : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                contraction = lambda;
                prefix      = "adaptive_contracted_" + std::to_string(moving) + "_" +
                              std::to_string(local) + "_" + std::to_string(lambda);
                input("cv_mean");
                ASSERT_TRUE(prepare(2));
                ASSERT_TRUE(attach("cv_mean"));
                ASSERT_TRUE(command("run 0"));
                const auto y      = coordinates(2);
                const auto force  = state();
                const auto energy = [&](const Positions &z) {
                    const auto q = contracted(physical(z, mobile, slab), lambda);
                    return local ? localOracle(q, "cv_mean") : oracle(q, "cv_mean");
                };
                EXPECT_NEAR(bias(), energy(y), 1.0e-11);
                for (int j = 0; j < 36; ++j) {
                    auto plus = y, minus = y;
                    plus[j] += 1.0e-5;
                    minus[j] -= 1.0e-5;
                    const double expected = -4 * (energy(plus) - energy(minus)) / 2.0e-5;
                    EXPECT_NEAR(force[27 * (j / 9) + 9 * ((j % 9) / 3) + 6 + j % 3], expected,
                                5.0e-8)
                        << prefix << " " << j;
                    ASSERT_TRUE(place(plus));
                    const double ep = bias();
                    ASSERT_TRUE(place(minus));
                    const double em = bias();
                    EXPECT_NEAR(-4 * (ep - em) / 2.0e-5, expected, 5.0e-8) << prefix;
                }
                ASSERT_TRUE(place(y));
                auto *pl = lmp->modify->get_fix_by_id("pl");
                std::array<double, 6> virial{};
                for (int k = 0; k < 6; ++k)
                    virial[k] = pl->virial[k] / 4;
                MPI_Allreduce(MPI_IN_PLACE, virial.data(), 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                const int row[6] = {0, 1, 2, 0, 0, 1}, col[6] = {0, 1, 2, 1, 2, 2};
                for (int k = 0; k < 6; ++k) {
                    auto plus = y, minus = y;
                    for (int b = 0; b < 4; ++b)
                        for (int i = 0; i < 3; ++i) {
                            plus[9 * b + 3 * i + col[k]] += 1.0e-5 * y[9 * b + 3 * i + row[k]];
                            minus[9 * b + 3 * i + col[k]] -= 1.0e-5 * y[9 * b + 3 * i + row[k]];
                        }
                    EXPECT_NEAR(virial[k], -(energy(plus) - energy(minus)) / 2.0e-5, 1.0e-7)
                        << prefix << " strain " << k;
                }
                close();
                clean();
            }
        }
    }
}

TEST_F(PlumedAdaptiveMPI, adaptive_probability_and_mixture_gradients)
{
    mobile = true;
    for (const std::string mode : {"probability", "mixture"}) {
        prefix = "adaptive_" + mode;
        input(mode);
        ASSERT_TRUE(prepare(2));
        ASSERT_TRUE(attach(mode));
        ASSERT_TRUE(command("run 0"));
        const auto y = coordinates(2), q = physical(y, true);
        const auto force = state();
        EXPECT_NEAR(bias(), oracle(q, mode), 1.0e-11);
        for (int j = 0; j < 36; ++j) {
            auto plus = y, minus = y;
            plus[j] += 1.0e-5;
            minus[j] -= 1.0e-5;
            const double expected =
                -4 * (oracle(physical(plus, true), mode) - oracle(physical(minus, true), mode)) /
                2.0e-5;
            EXPECT_NEAR(force[27 * (j / 9) + 9 * ((j % 9) / 3) + 6 + j % 3], expected, 5.0e-8);
        }
        close();
        clean();
    }
}

TEST_F(PlumedAdaptiveMPI, physical_outside_wall_complete_adaptive_gradients)
{
    localCV = outsideWall = true;
    for (bool moving : {false, true}) {
        mobile = moving;
        for (bool plane : {false, true}) {
            localSlab = slab = plane;
            prefix = "adaptive_physical_wall_" + std::to_string(moving) + std::to_string(plane);
            input("bias_mean");
            ASSERT_TRUE(prepare(2));
            ASSERT_TRUE(attach("bias_mean"));
            ASSERT_TRUE(command("run 0"));
            const auto y      = coordinates(2);
            const auto force  = state();
            const auto energy = [&](const Positions &z) {
                const auto q = physical(z, mobile, slab);
                double value = 0;
                for (int b = 0; b < 4; ++b) {
                    const double activity = localActivity(q.data() + 9 * b, true, localSlab);
                    value += 2 * activity * activity / 4;
                }
                return value;
            };
            EXPECT_NEAR(bias(), energy(y), 1.0e-11);
            for (int j = 0; j < 36; ++j) {
                auto plus = y, minus = y;
                plus[j] += 1.0e-5;
                minus[j] -= 1.0e-5;
                const double expected = -4 * (energy(plus) - energy(minus)) / 2.0e-5;
                EXPECT_NEAR(force[27 * (j / 9) + 9 * ((j % 9) / 3) + 6 + j % 3], expected, 5.0e-8);
                ASSERT_TRUE(place(plus));
                const double ep = bias();
                ASSERT_TRUE(place(minus));
                const double em = bias();
                EXPECT_NEAR(-4 * (ep - em) / 2.0e-5, expected, 5.0e-8);
            }
            close();
            clean();
        }
    }
}

TEST_F(PlumedAdaptiveMPI, physical_path_request_rejects_virtual_coordinates)
{
    for (const std::string mode : {"coordinate_mean", "cv_mean"}) {
        prefix = "adaptive_rejected_physical_view_" + mode;
        input(mode);
        ASSERT_TRUE(prepare(2));
        const std::string request = "fix pl all plumed plumedfile " + prefix + ".dat outfile " +
                                    prefix + ".log path_integral " + mode +
                                    " pimd_fix pi path_coordinates physical" +
                                    (mode == "cv_mean" ? " path_contraction 0.5" : "");
        lammps_commands_string(lmp, request.c_str());
        int failed = lammps_has_error(lmp);
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        EXPECT_EQ(failed, 1);
        char message[1024]{};
        lammps_get_last_error_message(lmp, message, sizeof(message));
        EXPECT_NE(std::string(message).find("requires uncontracted"), std::string::npos);
        close();
        clean();
    }
}
