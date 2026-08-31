// -*- C++ -*-
#ifndef _HYBRID_ENGINE_FIELD_HPP_
#define _HYBRID_ENGINE_FIELD_HPP_

#include "hybrid.hpp"
#include "nix/nix.hpp"

#include <array>
#include <stdexcept>

namespace hybrid::engine
{
using FieldState  = std::array<nix::float64, num_field_components>;
using VectorState = std::array<nix::float64, num_vector_components>;

struct GridSpacing {
  nix::float64 x;
  nix::float64 y;
  nix::float64 z;
};

inline nix::float64 magnetic_cell_to_face(int component, nix::float64 cell,
                                          nix::float64 forward_cell)
{
  if (component < field_component::magnetic_x || component > field_component::magnetic_z) {
    throw std::out_of_range("Hybrid magnetic interpolation component must be Bx, By, or Bz");
  }
  return 0.5 * (cell + forward_cell);
}

inline nix::float64 magnetic_face_to_cell(int component, nix::float64 face,
                                          nix::float64 backward_face)
{
  if (component < field_component::magnetic_x || component > field_component::magnetic_z) {
    throw std::out_of_range("Hybrid magnetic interpolation component must be Bx, By, or Bz");
  }
  return 0.5 * (face + backward_face);
}

inline nix::float64 edge_electric_average(nix::float64 value00, nix::float64 value01,
                                          nix::float64 value10, nix::float64 value11)
{
  return 0.25 * (value00 + value01 + value10 + value11);
}

inline VectorState curl_magnetic(const FieldState& x_plus, const FieldState& x_minus,
                                 const FieldState& y_plus, const FieldState& y_minus,
                                 const FieldState& z_plus, const FieldState& z_minus,
                                 const GridSpacing& spacing, nix::float64 light_speed)
{
  const nix::float64 cdx = 0.5 * light_speed / spacing.x / nix::math::pi4;
  const nix::float64 cdy = 0.5 * light_speed / spacing.y / nix::math::pi4;
  const nix::float64 cdz = 0.5 * light_speed / spacing.z / nix::math::pi4;
  return {cdy * (y_plus[field_component::magnetic_z] - y_minus[field_component::magnetic_z]) -
              cdz * (z_plus[field_component::magnetic_y] - z_minus[field_component::magnetic_y]),
          cdz * (z_plus[field_component::magnetic_x] - z_minus[field_component::magnetic_x]) -
              cdx * (x_plus[field_component::magnetic_z] - x_minus[field_component::magnetic_z]),
          cdx * (x_plus[field_component::magnetic_y] - x_minus[field_component::magnetic_y]) -
              cdy * (y_plus[field_component::magnetic_x] - y_minus[field_component::magnetic_x])};
}

inline VectorState constrained_transport_magnetic(
    const FieldState& baseline, const FieldState& edge_electric, const FieldState& x_minus_electric,
    const FieldState& y_minus_electric, const FieldState& z_minus_electric,
    const GridSpacing& spacing, nix::float64 light_speed, nix::float64 time_step)
{
  const nix::float64 cdtx = light_speed * time_step / spacing.x;
  const nix::float64 cdty = light_speed * time_step / spacing.y;
  const nix::float64 cdtz = light_speed * time_step / spacing.z;
  return {baseline[field_component::magnetic_x] -
              cdty * (edge_electric[field_component::electric_z] -
                      y_minus_electric[field_component::electric_z]) +
              cdtz * (edge_electric[field_component::electric_y] -
                      z_minus_electric[field_component::electric_y]),
          baseline[field_component::magnetic_y] -
              cdtz * (edge_electric[field_component::electric_x] -
                      z_minus_electric[field_component::electric_x]) +
              cdtx * (edge_electric[field_component::electric_z] -
                      x_minus_electric[field_component::electric_z]),
          baseline[field_component::magnetic_z] -
              cdtx * (edge_electric[field_component::electric_y] -
                      x_minus_electric[field_component::electric_y]) +
              cdty * (edge_electric[field_component::electric_x] -
                      y_minus_electric[field_component::electric_x])};
}
} // namespace hybrid::engine

#endif
