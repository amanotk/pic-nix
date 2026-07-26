// -*- C++ -*-
#ifndef _HYBRID_ENGINE_INTERPOLATION_HPP_
#define _HYBRID_ENGINE_INTERPOLATION_HPP_

#include "nix/array_types.hpp"
#include "nix/xtensor/xtensor_particle.hpp"

#include <array>
#include <cmath>

namespace hybrid::engine
{
using Position   = std::array<nix::float64, 3>;
using GridIndex  = std::array<int, 3>;
using FieldValue = std::array<nix::float64, 6>;

inline nix::float64 shape2(nix::float64 particle_position, nix::float64 grid_position,
                           nix::float64 reciprocal_spacing)
{
  const nix::float64 distance = std::abs((particle_position - grid_position) * reciprocal_spacing);
  if (distance <= 0.5) {
    return 0.75 - distance * distance;
  }
  if (distance <= 1.5) {
    const nix::float64 edge_distance = 1.5 - distance;
    return 0.5 * edge_distance * edge_distance;
  }
  return 0;
}

inline GridIndex particle_cell(const nix::XtensorParticle& particle, const Position& position)
{
  const int ix = particle.has_xdim
                     ? particle.Lbx + static_cast<int>(
                                          std::floor((position[0] - particle.xmin) / particle.delx))
                     : particle.Lbx;
  const int iy = particle.has_ydim
                     ? particle.Lby + static_cast<int>(
                                          std::floor((position[1] - particle.ymin) / particle.dely))
                     : particle.Lby;
  const int iz = particle.has_zdim
                     ? particle.Lbz + static_cast<int>(
                                          std::floor((position[2] - particle.zmin) / particle.delz))
                     : particle.Lbz;
  return {iz, iy, ix};
}

inline FieldValue interpolate_collocated(const nix::Array4D<nix::float64>& field,
                                         const nix::Array4D<nix::float64>& background,
                                         const nix::XtensorParticle&       particle,
                                         const GridIndex& anchor, const Position& position)
{
  std::array<nix::float64, 5> weight_x = {};
  std::array<nix::float64, 5> weight_y = {};
  std::array<nix::float64, 5> weight_z = {};

  for (int offset = -2; offset <= 2; ++offset) {
    const int slot = offset + 2;
    if (particle.has_xdim) {
      const nix::float64 coordinate =
          particle.xmin + particle.delx * (anchor[2] + offset - particle.Lbx + 0.5);
      weight_x[slot] = shape2(position[0], coordinate, 1.0 / particle.delx);
    }
    if (particle.has_ydim) {
      const nix::float64 coordinate =
          particle.ymin + particle.dely * (anchor[1] + offset - particle.Lby + 0.5);
      weight_y[slot] = shape2(position[1], coordinate, 1.0 / particle.dely);
    }
    if (particle.has_zdim) {
      const nix::float64 coordinate =
          particle.zmin + particle.delz * (anchor[0] + offset - particle.Lbz + 0.5);
      weight_z[slot] = shape2(position[2], coordinate, 1.0 / particle.delz);
    }
  }
  if (!particle.has_xdim) {
    weight_x[2] = 1;
  }
  if (!particle.has_ydim) {
    weight_y[2] = 1;
  }
  if (!particle.has_zdim) {
    weight_z[2] = 1;
  }

  FieldValue result = {};
  for (int jz = 0; jz < 5; ++jz) {
    for (int jy = 0; jy < 5; ++jy) {
      for (int jx = 0; jx < 5; ++jx) {
        const nix::float64 weight = weight_x[jx] * weight_y[jy] * weight_z[jz];
        if (weight == 0) {
          continue;
        }
        const int iz = anchor[0] + jz - 2;
        const int iy = anchor[1] + jy - 2;
        const int ix = anchor[2] + jx - 2;
        for (int component = 0; component < 3; ++component) {
          result[component] += weight * field(iz, iy, ix, component);
          result[component + 3] +=
              weight * (field(iz, iy, ix, component + 3) + background(iz, iy, ix, component));
        }
      }
    }
  }
  return result;
}
} // namespace hybrid::engine

#endif
