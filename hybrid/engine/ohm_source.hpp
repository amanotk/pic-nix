// -*- C++ -*-
#ifndef _HYBRID_ENGINE_OHM_SOURCE_HPP_
#define _HYBRID_ENGINE_OHM_SOURCE_HPP_

#include "hybrid.hpp"
#include "nix/nix.hpp"

#include <array>
#include <cmath>

namespace hybrid::engine
{
using OhmSource   = std::array<nix::float64, num_ohm_source_components>;
using VectorState = std::array<nix::float64, num_vector_components>;

inline OhmSource
construct_ohm_source(const std::array<nix::float64, num_moment_components>& moment,
                     const std::array<nix::float64, num_field_components>&  field,
                     const VectorState& background, nix::float64 light_speed,
                     nix::float64 spacing_x, nix::float64 spacing_y, nix::float64 spacing_z,
                     const std::array<nix::float64, num_moment_components>& plus_x_plus,
                     const std::array<nix::float64, num_moment_components>& plus_x_minus,
                     const std::array<nix::float64, num_moment_components>& plus_y_plus,
                     const std::array<nix::float64, num_moment_components>& plus_y_minus,
                     const std::array<nix::float64, num_moment_components>& plus_z_plus,
                     const std::array<nix::float64, num_moment_components>& plus_z_minus)
{
  const nix::float64 reciprocal_light_speed = 1.0 / light_speed;
  const nix::float64 rdx                    = 0.5 / spacing_x;
  const nix::float64 rdy                    = 0.5 / spacing_y;
  const nix::float64 rdz                    = 0.5 / spacing_z;

  const auto Bx = field[field_component::magnetic_x] + background[0];
  const auto By = field[field_component::magnetic_y] + background[1];
  const auto Bz = field[field_component::magnetic_z] + background[2];

  OhmSource result                  = {};
  result[current_component::charge] = moment[moment_component::density];
  result[current_component::current_x] =
      -(moment[moment_component::momentum_y] * Bz - moment[moment_component::momentum_z] * By) *
      reciprocal_light_speed;
  result[current_component::current_y] =
      -(moment[moment_component::momentum_z] * Bx - moment[moment_component::momentum_x] * Bz) *
      reciprocal_light_speed;
  result[current_component::current_z] =
      -(moment[moment_component::momentum_x] * By - moment[moment_component::momentum_y] * Bx) *
      reciprocal_light_speed;

  result[current_component::current_x] +=
      rdx * (plus_x_plus[moment_component::stress_xx] - plus_x_minus[moment_component::stress_xx]) +
      rdy * (plus_y_plus[moment_component::stress_xy] - plus_y_minus[moment_component::stress_xy]) +
      rdz * (plus_z_plus[moment_component::stress_xz] - plus_z_minus[moment_component::stress_xz]);
  result[current_component::current_y] +=
      rdx * (plus_x_plus[moment_component::stress_xy] - plus_x_minus[moment_component::stress_xy]) +
      rdy * (plus_y_plus[moment_component::stress_yy] - plus_y_minus[moment_component::stress_yy]) +
      rdz * (plus_z_plus[moment_component::stress_yz] - plus_z_minus[moment_component::stress_yz]);
  result[current_component::current_z] +=
      rdx * (plus_x_plus[moment_component::stress_xz] - plus_x_minus[moment_component::stress_xz]) +
      rdy * (plus_y_plus[moment_component::stress_yz] - plus_y_minus[moment_component::stress_yz]) +
      rdz * (plus_z_plus[moment_component::stress_zz] - plus_z_minus[moment_component::stress_zz]);

  return result;
}

inline VectorState resistive_field(const VectorState& rot, const VectorState& rot_x_minus,
                                   const VectorState& rot_x_plus, const VectorState& rot_y_minus,
                                   const VectorState& rot_y_plus, const VectorState& rot_z_minus,
                                   const VectorState& rot_z_plus, nix::float64 eta,
                                   nix::float64 chi, nix::float64 spacing_x, nix::float64 spacing_y,
                                   nix::float64 spacing_z)
{
  const nix::float64 rdx2 = 1.0 / (spacing_x * spacing_x);
  const nix::float64 rdy2 = 1.0 / (spacing_y * spacing_y);
  const nix::float64 rdz2 = 1.0 / (spacing_z * spacing_z);

  VectorState result = {};
  for (int direction = 0; direction < num_vector_components; ++direction) {
    result[direction] =
        eta * rot[direction] -
        chi * ((rot_x_minus[direction] - 2 * rot[direction] + rot_x_plus[direction]) * rdx2 +
               (rot_y_minus[direction] - 2 * rot[direction] + rot_y_plus[direction]) * rdy2 +
               (rot_z_minus[direction] - 2 * rot[direction] + rot_z_plus[direction]) * rdz2);
  }
  return result;
}
} // namespace hybrid::engine

#endif
