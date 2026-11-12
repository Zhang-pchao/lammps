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

#ifndef LMP_PIMD_ADAPTIVE_MAP_H
#define LMP_PIMD_ADAPTIVE_MAP_H

#include <cmath>
#include <stdexcept>

namespace LAMMPS_NS {
namespace PIMDAdaptiveMap {

  // Input displacement is already minimum-imaged by the owning domain.
  inline double amplitude(const double *delta, double core, double outer, double *gradient,
                          int axis = -1)
  {
    if (!std::isfinite(core) || !std::isfinite(outer) || (core < 0.0) || (outer <= core))
      throw std::invalid_argument("Adaptive radii require 0 <= core < outer");
    if ((axis < -1) || (axis > 2))
      throw std::invalid_argument("Adaptive slab axis must be x, y or z");
    double r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(delta[d]))
        throw std::invalid_argument("Adaptive coordinates must be finite");
      gradient[d] = 0.0;
      if ((axis < 0) || (axis == d)) r2 += delta[d] * delta[d];
    }
    const double r = std::sqrt(r2);
    if (r <= core) return 1.0;
    if (r >= outer) return 0.0;
    const double t = (r - core) / (outer - core);
    const double s = 1.0 - t;
    // Factored value is stable at the classical endpoint.
    const double a = s * s * s * (1.0 + 3.0 * t + 6.0 * t * t);
    const double derivative = -30.0 * t * t * s * s / ((outer - core) * r);
    for (int d = 0; d < 3; ++d)
      if ((axis < 0) || (axis == d)) gradient[d] = derivative * delta[d];
    return a;
  }

  inline double force(double a, double physical, double mean_force, double gradient,
                      double mean_force_offset)
  {
    // The native integrator samples P*U at P*T. No additional 1/P belongs here.
    return a * physical + (1.0 - a) * mean_force + gradient * mean_force_offset;
  }

  inline double velocity(double a, double auxiliary, double mean_velocity, double offset,
                         double gradient_dot_mean_velocity)
  { return a * auxiliary + (1.0 - a) * mean_velocity + offset * gradient_dot_mean_velocity; }
}    // namespace PIMDAdaptiveMap
}    // namespace LAMMPS_NS
#endif
