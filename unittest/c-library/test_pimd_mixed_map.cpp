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

#include "REPLICA/pimd_mixed_map.h"

#include <array>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

using LAMMPS_NS::PIMDMixedMap;

TEST(PIMDMixedMap, rejects_invalid_counts_and_indices)
{
    for (const auto &counts :
         std::vector<std::array<int, 2>>{{0, 1}, {8, 0}, {8, -1}, {8, 3}, {3, 1}, {4, 8}})
        EXPECT_THROW(PIMDMixedMap(counts[0], counts[1]), std::invalid_argument);
    PIMDMixedMap map(8, 2);
    EXPECT_THROW(map.bead_index(-1), std::out_of_range);
    EXPECT_THROW(map.bead_index(8), std::out_of_range);
    EXPECT_THROW(map.representative(2), std::out_of_range);
    EXPECT_THROW(map.coefficient(2, 0), std::out_of_range);
    EXPECT_THROW(map.eigenvalue(-1), std::out_of_range);
    EXPECT_THROW(map.physical_force(nullptr, 1, 0), std::invalid_argument);
    const double value = 1.0;
    EXPECT_THROW(map.physical_force(&value, 0, 0), std::invalid_argument);
}

TEST(PIMDMixedMap, contiguous_nested_buckets_and_unique_owners)
{
    const PIMDMixedMap map(8, 2);
    const int expected[] = {0, 0, 0, 0, 1, 1, 1, 1};
    int owners           = 0;
    for (int b = 0; b < 8; ++b) {
        EXPECT_EQ(map.bead_index(b), expected[b]);
        EXPECT_EQ(map.is_representative(b), (b == 0) || (b == 4));
        owners += map.is_representative(b);
    }
    EXPECT_EQ(owners, 2);
    EXPECT_EQ(map.copies_per_bead(), 4);
    EXPECT_EQ(map.representative(0), 0);
    EXPECT_EQ(map.representative(1), 4);
    const PIMDMixedMap large(1 << 30, 1 << 29);
    EXPECT_EQ(large.bead_index((1 << 30) - 1), (1 << 29) - 1);
    EXPECT_EQ(large.representative((1 << 29) - 1), (1 << 30) - 2);
}

TEST(PIMDMixedMap, classical_and_uniform_limits)
{
    const double f[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const PIMDMixedMap classical(8, 1), uniform(8, 8), single(1, 1);
    EXPECT_DOUBLE_EQ(classical.physical_force(f, 1, 0), 4.5);
    EXPECT_DOUBLE_EQ(single.physical_force(f, 1, 0), 1.0);
    for (int b = 0; b < 8; ++b) {
        EXPECT_EQ(classical.bead_index(b), 0);
        EXPECT_EQ(uniform.bead_index(b), b);
        EXPECT_DOUBLE_EQ(uniform.physical_force(f, 1, b), f[b] / 8.0);
    }
}

TEST(PIMDMixedMap, weighted_adjoint_and_mass_conversion)
{
    for (int p : {1, 2, 4, 8}) {
        const PIMDMixedMap map(8, p);
        double left = 0.0, right = 0.0;
        std::array<double, 24> f{};
        std::vector<double> q(p);
        for (int a = 0; a < p; ++a)
            q[a] = std::sin(a + 0.7);
        for (int b = 0; b < 8; ++b) {
            f[3 * b + 1] = std::cos(0.3 * b);
            left += f[3 * b + 1] * q[map.bead_index(b)] / 8.0;
        }
        for (int a = 0; a < p; ++a) {
            const double force = map.physical_force(f.data() + 1, 3, a);
            right += force * q[a];
            double average = 0.0;
            for (int b = a * (8 / p); b < (a + 1) * (8 / p); ++b)
                average += f[3 * b + 1] / (8 / p);
            EXPECT_NEAR(p * force, average, 1.0e-14);
        }
        EXPECT_NEAR(left, right, 1.0e-14);
    }
}

namespace {
// Three coupled scalar atomic coordinates with a nonlinear three-body term.
// This energy is evaluated on full configurations; the derivative is analytic.
double potential(const std::array<double, 3> &x, std::array<double, 3> &f)
{
    const double a = x[0] - x[1], b = x[1] - x[2];
    const double t = a * b;
    double energy  = 0.07 * t * t;
    for (int i = 0; i < 3; ++i) {
        energy += 0.5 * (i + 1) * x[i] * x[i];
        f[i] = -(i + 1) * x[i];
    }
    f[0] -= 0.14 * t * b;
    f[1] -= 0.14 * t * (a - b);
    f[2] += 0.14 * t * a;
    return energy;
}

double lifted_energy(const std::vector<PIMDMixedMap> &maps,
                     const std::vector<std::vector<double>> &q, std::vector<double> &raw)
{
    const int P = maps[0].partitions();
    raw.resize(3 * P);
    double energy = 0.0;
    for (int b = 0; b < P; ++b) {
        std::array<double, 3> x{}, f{};
        for (int i = 0; i < 3; ++i)
            x[i] = q[i][maps[i].bead_index(b)];
        energy += potential(x, f) / P;
        for (int i = 0; i < 3; ++i)
            raw[3 * b + i] = f[i];
    }
    return energy;
}
} // namespace

TEST(PIMDMixedMap, nonlinear_manybody_physical_gradient)
{
    for (const auto &counts :
         std::vector<std::array<int, 3>>{{1, 1, 1}, {8, 8, 8}, {8, 1, 1}, {8, 2, 1}, {32, 4, 1}}) {
        const int P = counts[0];
        std::vector<PIMDMixedMap> maps;
        std::vector<std::vector<double>> q(3);
        for (int i = 0; i < 3; ++i) {
            maps.emplace_back(P, counts[i]);
            q[i].resize(counts[i]);
            for (int a = 0; a < counts[i]; ++a)
                q[i][a] = std::sin(0.3 * a + i + 0.7);
        }
        std::vector<double> raw, scratch;
        lifted_energy(maps, q, raw);
        for (int i = 0; i < 3; ++i) {
            for (int a = 0; a < counts[i]; ++a) {
                const double force = maps[i].physical_force(raw.data() + i, 3, a);
                for (double h : {1.0e-4, 3.0e-5, 1.0e-5}) {
                    const double saved = q[i][a];
                    q[i][a]            = saved + h;
                    const double plus  = lifted_energy(maps, q, scratch);
                    q[i][a]            = saved - h;
                    const double minus = lifted_energy(maps, q, scratch);
                    q[i][a]            = saved;
                    EXPECT_NEAR(force, -(plus - minus) / (2 * h), 1.0e-7);
                }
            }
        }
    }
}

TEST(PIMDMixedMap, orthonormal_cyclic_modes)
{
    for (int p : {1, 2, 4, 8, 32}) {
        const PIMDMixedMap map(p, p);
        for (int k = 0; k < p; ++k) {
            for (int l = 0; l < p; ++l) {
                double product = 0.0;
                for (int a = 0; a < p; ++a)
                    product += map.coefficient(k, a) * map.coefficient(l, a);
                EXPECT_NEAR(product, k == l ? 1.0 : 0.0, 1.0e-12);
            }
            for (int a = 0; a < p; ++a) {
                const double laplacian = 2 * map.coefficient(k, a) -
                                         map.coefficient(k, (a + 1) % p) -
                                         map.coefficient(k, (a + p - 1) % p);
                EXPECT_NEAR(laplacian, map.eigenvalue(k) * map.coefficient(k, a), 1.0e-12);
            }
        }
    }
}

TEST(PIMDMixedMap, spring_energy_is_diagonal_in_unique_modes)
{
    for (int p : {1, 2, 4, 8, 32}) {
        const PIMDMixedMap map(32, p);
        std::vector<double> q(p);
        for (int a = 0; a < p; ++a)
            q[a] = 0.1 * a + std::cos(a + 0.4);
        double direct = 0.0, modes = 0.0;
        for (int a = 0; a < p; ++a) {
            const double delta = q[a] - q[(a + 1) % p];
            direct += delta * delta;
        }
        for (int k = 0; k < p; ++k) {
            double mode = 0.0;
            for (int a = 0; a < p; ++a)
                mode += map.coefficient(k, a) * q[a];
            modes += map.eigenvalue(k) * mode * mode;
        }
        EXPECT_NEAR(direct, modes, 1.0e-12);
    }
}
