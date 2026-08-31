// -*- C++ -*-
#ifndef _HYBRID_ENGINE_MOMENT_HPP_
#define _HYBRID_ENGINE_MOMENT_HPP_

#include "interpolation.hpp"

#include "hybrid_chunk.hpp"

#include <array>
#include <stdexcept>

namespace hybrid::engine
{
inline void deposit_moments(HybridChunk::DataContainer& data)
{
  data.moment_kinetic.fill(0);

  for (int species = 0; species < data.num_species; ++species) {
    auto& particle = *data.particles[species];
    for (int ip = 0; ip < particle.Np; ++ip) {
      const Position  position = {particle.xu(ip, 0), particle.xu(ip, 1), particle.xu(ip, 2)};
      const GridIndex anchor   = particle_cell(particle, position);
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

      const nix::float64                                    vx    = particle.xu(ip, 3);
      const nix::float64                                    vy    = particle.xu(ip, 4);
      const nix::float64                                    vz    = particle.xu(ip, 5);
      const std::array<nix::float64, num_moment_components> value = {
          1, vx, vy, vz, vx * vx, vy * vy, vz * vz, vx * vy, vx * vz, vy * vz};
      for (int jz = 0; jz < 5; ++jz) {
        for (int jy = 0; jy < 5; ++jy) {
          for (int jx = 0; jx < 5; ++jx) {
            const nix::float64 weight = particle.m * weight_x[jx] * weight_y[jy] * weight_z[jz];
            if (weight == 0) {
              continue;
            }
            for (int component = 0; component < num_moment_components; ++component) {
              data.moment_kinetic(anchor[0] + jz - 2, anchor[1] + jy - 2, anchor[2] + jx - 2,
                                  species, component) += weight * value[component];
            }
          }
        }
      }
    }
  }
}

inline void derive_current(HybridChunk::DataContainer& data)
{
  data.current_kinetic.fill(0);
  for (int iz = 0; iz < static_cast<int>(data.current_kinetic.shape()[0]); ++iz) {
    for (int iy = 0; iy < static_cast<int>(data.current_kinetic.shape()[1]); ++iy) {
      for (int ix = 0; ix < static_cast<int>(data.current_kinetic.shape()[2]); ++ix) {
        for (int species = 0; species < data.num_species; ++species) {
          const auto& particle = *data.particles[species];
          if (particle.m == 0) {
            throw std::invalid_argument("Hybrid current conversion requires nonzero particle mass");
          }
          const nix::float64 charge_to_mass = particle.q / particle.m;
          for (int component = 0; component < num_current_components; ++component) {
            data.current_kinetic(iz, iy, ix, component) +=
                charge_to_mass * data.moment_kinetic(iz, iy, ix, species, component);
          }
        }
      }
    }
  }
}
} // namespace hybrid::engine

#endif
