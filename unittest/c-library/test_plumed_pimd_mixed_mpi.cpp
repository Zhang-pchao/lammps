// Actual PLUMED gradients and shared-history continuity with mixed beads.
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
double localActivity(const double *q)
{
  double d12 = 0, d32 = 0, radius = 0;
  for (int d = 0; d < 3; ++d) {
    d12 += std::pow(q[d] - q[3 + d], 2);
    d32 += std::pow(q[6 + d] - q[3 + d], 2);
    radius += std::pow(q[d] - q[6 + d], 2);
  }
  const double first = std::exp(-1.2 * std::sqrt(d12));
  const double other = std::exp(-1.2 * std::sqrt(d32));
  const double defect = first / (first + other) - 0.5;
  const double t = std::max(0.0, std::min(1.0, (std::sqrt(radius) - 1.5) / 1.3));
  const double chi = 1 - 10 * std::pow(t, 3) + 15 * std::pow(t, 4) - 6 * std::pow(t, 5);
  return (1 + chi) * defect * defect;
}

double localOracle(const Positions &q, const std::string &mode)
{
  if (mode == "coordinate_mean") {
    std::array<double, 9> centroid{};
    for (int b = 0; b < 4; ++b)
      for (int i = 0; i < 9; ++i) centroid[i] += q[9 * b + i] / 4;
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

class PlumedMixedMPI : public ::testing::Test {
protected:
    bool localCV = false;
    LAMMPS *lmp        = nullptr;
    int rank           = 0;
    std::string prefix = "mixed_plumed";
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
        LAMMPS::argv args = {"mixed_plumed", "-screen", "none", "-log", "none",
                             "-partition",   "4x2",     "-in",  "none", "-nocite"};
        lmp               = new LAMMPS(args, MPI_COMM_WORLD);
        lammps_set_show_error(lmp, 0);
    }
    bool prepare(int coarse, bool restart = false)
    {
        open();
        if (restart) {
            if (!command(
                    "processors 2 1 1\nvariable file world mixed_plumed.0 mixed_plumed.1 "
                    "mixed_plumed.2 mixed_plumed.3\nread_restart ${file}\natom_modify sort 0 0"))
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
        return command("timestep 0.0001\nthermo 100000\nfix pi all pimd/mixed temp 1 sp "
                       "6.283185307179586 beads fine 4 beads coarse " +
                       std::to_string(coarse));
    }
    void input(const std::string &mode, bool metad = false, bool restart = false)
    {
        if (rank == 0) {
            std::ofstream out(prefix + ".dat");
            if (restart) out << "RESTART\n";
            if (localCV)
                out << "s: VORONOI_LOCAL_COORDINATION CENTERS=1,3 ASSIGNED=2 "
                       "REGION_CENTER=3 REFERENCE=0.5 KAPPA=1.2 CORE_RADIUS=1.5 OUTER_RADIUS=2.8 "
                       "PAIR_SWITCH=2.5 PAIR_CUTOFF=3.5 SKIN=0.2 NL_STRIDE=10 NOPBC\n";
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
                       " pimd_fix pi");
    }
    double bias() { return lmp->modify->get_fix_by_id("pi")->compute_vector(6); }
    std::array<double, 108> state()
    {
        std::array<double, 108> a{};
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const int j = 27 * (rank / 2) + 9 * (lmp->atom->tag[i] - 1);
            for (int d = 0; d < 3; ++d) {
                a[j + d]     = lmp->atom->x[i][d];
                a[j + 3 + d] = lmp->atom->v[i][d];
                a[j + 6 + d] = lmp->atom->f[i][d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, a.data(), a.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return a;
    }
    bool place(const Positions &q)
    {
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            for (int d = 0; d < 3; ++d)
                lmp->atom->x[i][d] = q[9 * (rank / 2) + 3 * (lmp->atom->tag[i] - 1) + d];
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
                std::remove(("mixed_plumed." + std::to_string(b)).c_str());
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

TEST_F(PlumedMixedMPI, nonlinear_modes_and_complete_path_finite_differences)
{
    for (int coarse : {1, 2})
        for (const std::string mode :
             {"coordinate_mean", "cv_mean", "bias_mean", "probability", "mixture"}) {
            SCOPED_TRACE(mode + ":" + std::to_string(coarse));
            prefix = "mixed_plumed_static_" + mode + std::to_string(coarse);
            input(mode);
            ASSERT_TRUE(prepare(coarse));
            ASSERT_TRUE(attach(mode));
            ASSERT_TRUE(command("run 0"));
            const auto q      = coordinates(coarse);
            const auto force  = state();
            const double base = bias();
            EXPECT_NEAR(base, oracle(q, mode), 1e-11);
            auto *pi = lmp->modify->get_fix_by_id("pi");
            EXPECT_NEAR(pi->compute_vector(0), 0, 1e-11);
            EXPECT_NEAR(pi->compute_vector(5), 4.5, 1e-11);
            for (int i = 0; i < 3; ++i) {
                const int p = i == 0 ? 4 : (i == 1 ? coarse : 1);
                for (int a = 0; a < p; ++a)
                    for (int d = 0; d < 3; ++d) {
                        for (double delta : {1e-4, 1e-5, 1e-6}) {
                            auto plus = q, minus = q;
                            for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b) {
                                plus[9 * b + 3 * i + d] += delta;
                                minus[9 * b + 3 * i + d] -= delta;
                            }
                            ASSERT_TRUE(place(plus));
                            const double ep = bias();
                            ASSERT_TRUE(place(minus));
                            const double em         = bias();
                            const double derivative = -p * (ep - em) / (2 * delta);
                            for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b)
                                EXPECT_NEAR(force[27 * b + 9 * i + 6 + d], derivative, 5e-7);
                        }
                    }
            }
            ASSERT_TRUE(place(q));
            ASSERT_TRUE(command("fix_modify pl energy no\nrun 0"));
            EXPECT_NEAR(bias(), base, 1e-11);
            EXPECT_NEAR(pi->compute_vector(0), 0, 1e-11);
            EXPECT_NEAR(pi->compute_scalar(), pi->compute_vector(1) + pi->compute_vector(2) + base,
                        1e-11);
            close();
            clean();
        }
}

TEST_F(PlumedMixedMPI, normalized_metad_history_repeated_step_and_restart)
{
    for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
        SCOPED_TRACE(mode);
        prefix = "mixed_plumed_history_" + mode;
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
        ASSERT_TRUE(command("variable file world mixed_plumed.0 mixed_plumed.1 mixed_plumed.2 "
                            "mixed_plumed.3\nwrite_restart ${file}"));
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
TEST_F(PlumedMixedMPI, local_voronoi_modes_and_complete_path_finite_differences)
{
  localCV = true;
  for (int coarse : {1, 2})
    for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
      SCOPED_TRACE(mode + ":" + std::to_string(coarse));
      prefix = "mixed_local_static_" + mode + std::to_string(coarse);
      input(mode);
      ASSERT_TRUE(prepare(coarse));
      ASSERT_TRUE(attach(mode));
      ASSERT_TRUE(command("run 0"));
      const auto q = coordinates(coarse);
      const auto force = state();
      const double base = bias();
      EXPECT_NEAR(base, localOracle(q, mode), 1e-11);
      auto *pi = lmp->modify->get_fix_by_id("pi");
      EXPECT_NEAR(pi->compute_vector(0), 0, 1e-11);
      EXPECT_NEAR(pi->compute_vector(5), 4.5, 1e-11);
      for (int i = 0; i < 3; ++i) {
        const int p = i == 0 ? 4 : (i == 1 ? coarse : 1);
        for (int a = 0; a < p; ++a)
          for (int d = 0; d < 3; ++d) {
            for (double delta : {1e-4, 1e-5, 1e-6}) {
              auto plus = q, minus = q;
              for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b) {
                plus[9 * b + 3 * i + d] += delta;
                minus[9 * b + 3 * i + d] -= delta;
              }
              ASSERT_TRUE(place(plus));
              const double ep = bias();
              ASSERT_TRUE(place(minus));
              const double em = bias();
              const double derivative = -p * (ep - em) / (2 * delta);
              for (int b = a * 4 / p; b < (a + 1) * 4 / p; ++b)
                EXPECT_NEAR(force[27 * b + 9 * i + 6 + d], derivative, 5e-7);
            }
          }
      }
      ASSERT_TRUE(place(q));
      ASSERT_TRUE(command("fix_modify pl energy no\nrun 0"));
      EXPECT_NEAR(bias(), base, 1e-11);
      EXPECT_NEAR(pi->compute_vector(0), 0, 1e-11);
      EXPECT_NEAR(pi->compute_scalar(), pi->compute_vector(1) + pi->compute_vector(2) + base,
                  1e-11);
      close();
      clean();
    }
}

TEST_F(PlumedMixedMPI, local_voronoi_shared_history_and_restart)
{
  localCV = true;
  for (const std::string mode : {"coordinate_mean", "cv_mean", "bias_mean"}) {
    SCOPED_TRACE(mode);
    prefix = "mixed_local_history_" + mode;
    clean();
    input(mode, true);
    ASSERT_TRUE(prepare(2));
    ASSERT_TRUE(attach(mode));
    ASSERT_TRUE(command("run 2\nrun 0"));
    const auto before = state();
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
    for (int i = 0; i < 108; ++i) EXPECT_NEAR(repeated[i], before[i], 1e-12);
    ASSERT_TRUE(command("variable file world mixed_plumed.0 mixed_plumed.1 mixed_plumed.2 "
                        "mixed_plumed.3\nwrite_restart ${file}"));
    close();
    input(mode, true, true);
    ASSERT_TRUE(prepare(2, true));
    ASSERT_TRUE(attach(mode));
    ASSERT_TRUE(command("run 0"));
    const auto restored = state();
    EXPECT_NEAR(bias(), initial, 1e-10);
    for (int i = 0; i < 108; ++i) EXPECT_NEAR(restored[i], before[i], 1e-10);
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
