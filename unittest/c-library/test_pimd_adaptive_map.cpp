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

#include "REPLICA/pimd_adaptive_map.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace {
using namespace LAMMPS_NS::PIMDAdaptiveMap;
using Vec = std::array<double, 3>;

std::vector<Vec> coordinates(const std::vector<Vec> &y, Vec &c, Vec &gradient, double &a)
{
    c = {};
    for (const auto &b : y)
        for (int d = 0; d < 3; ++d)
            c[d] += b[d] / y.size();
    a      = amplitude(c.data(), 0.5, 2.5, gradient.data());
    auto q = y;
    for (auto &b : q)
        for (int d = 0; d < 3; ++d)
            b[d] = c[d] + a * (b[d] - c[d]);
    return q;
}

double energy(const std::vector<Vec> &y)
{
    Vec c, g;
    double a;
    const auto q = coordinates(y, c, g, a);
    double e     = 0.0;
    for (const auto &b : q)
        e += (0.5 * (b[0] * b[0] + 2 * b[1] * b[1] + 3 * b[2] * b[2]) +
              0.2 * std::sin(b[0] * b[1] * b[2])) /
             y.size();
    return e;
}
} // namespace

TEST(PIMDAdaptiveMap, endpoints_and_invalid_radii)
{
    Vec g, r{};
    EXPECT_DOUBLE_EQ(amplitude(r.data(), 0.5, 2.5, g.data()), 1.0);
    for (double v : g)
        EXPECT_DOUBLE_EQ(v, 0.0);
    r[0] = 2.5;
    EXPECT_DOUBLE_EQ(amplitude(r.data(), 0.5, 2.5, g.data()), 0.0);
    for (double v : g)
        EXPECT_DOUBLE_EQ(v, 0.0);
    EXPECT_THROW(amplitude(r.data(), 2.5, 0.5, g.data()), std::invalid_argument);
    EXPECT_THROW(amplitude(r.data(), -1.0, 0.5, g.data()), std::invalid_argument);
    r[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(amplitude(r.data(), 0.5, 2.5, g.data()), std::invalid_argument);
}

TEST(PIMDAdaptiveMap, independent_energy_and_velocity_derivatives)
{
    for (int p : {1, 2, 4, 8, 16}) {
        std::vector<Vec> y(p), v(p), f(p);
        for (int b = 0; b < p; ++b)
            for (int d = 0; d < 3; ++d) {
                y[b][d] = 0.7 + 0.2 * std::sin(b + 0.5 * d);
                v[b][d] = 0.1 * std::cos(b + 0.7 * d);
            }
        Vec c, g, fm{}, vm{};
        double a, moment = 0.0;
        const auto q = coordinates(y, c, g, a);
        for (int b = 0; b < p; ++b)
            for (int d = 0; d < 3; ++d) {
                f[b][d] = -(d + 1) * q[b][d] - 0.2 * std::cos(q[b][0] * q[b][1] * q[b][2]) *
                                                   q[b][(d + 1) % 3] * q[b][(d + 2) % 3];
                fm[d] += f[b][d] / p;
                vm[d] += v[b][d] / p;
                moment += f[b][d] * (y[b][d] - c[d]) / p;
            }
        double rate = 0.0;
        for (int d = 0; d < 3; ++d)
            rate += g[d] * vm[d];
        constexpr double h = 1.0e-5;
        for (int b = 0; b < p; ++b)
            for (int d = 0; d < 3; ++d) {
                auto plus = y, minus = y;
                plus[b][d] += h;
                minus[b][d] -= h;
                const double numerical = -(energy(plus) - energy(minus)) / (2 * h);
                EXPECT_NEAR(force(a, f[b][d], fm[d], g[d], moment) / p, numerical, 2.0e-9);
            }
        auto plus = y, minus = y;
        for (int b = 0; b < p; ++b)
            for (int d = 0; d < 3; ++d) {
                plus[b][d] += h * v[b][d];
                minus[b][d] -= h * v[b][d];
            }
        Vec unused, unused_gradient;
        double unused_a;
        const auto qp = coordinates(plus, unused, unused_gradient, unused_a);
        const auto qm = coordinates(minus, unused, unused_gradient, unused_a);
        for (int b = 0; b < p; ++b)
            for (int d = 0; d < 3; ++d)
                EXPECT_NEAR(velocity(a, v[b][d], vm[d], y[b][d] - c[d], rate),
                            (qp[b][d] - qm[b][d]) / (2 * h), 2.0e-10);
    }
}

TEST(PIMDAdaptiveMap, slab_transverse_invariance_and_gradient_sweep)
{
    for (int axis = 0; axis < 3; ++axis) {
        Vec delta{13.0, -8.0, 21.0}, gradient{};
        delta[axis]    = -1.2;
        const double a = amplitude(delta.data(), 0.5, 2.5, gradient.data(), axis);
        Vec radial{-1.2, 0, 0}, unused{};
        EXPECT_DOUBLE_EQ(a, amplitude(radial.data(), 0.5, 2.5, unused.data()));
        for (int d = 0; d < 3; ++d) {
            if (d != axis) EXPECT_DOUBLE_EQ(gradient[d], 0.0);
            for (double h : {1.0e-4, 1.0e-5, 1.0e-6}) {
                auto plus = delta, minus = delta;
                plus[d] += h;
                minus[d] -= h;
                const double numeric = (amplitude(plus.data(), 0.5, 2.5, unused.data(), axis) -
                                        amplitude(minus.data(), 0.5, 2.5, unused.data(), axis)) /
                                       (2 * h);
                EXPECT_NEAR(gradient[d], numeric, 2.0e-8);
            }
        }
    }
    Vec r{}, g{};
    EXPECT_THROW(amplitude(r.data(), 0.5, 2.5, g.data(), 3), std::invalid_argument);
}
