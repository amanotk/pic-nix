// -*- C++ -*-
#ifndef _HYBRID_ENGINE_FILTER_HPP_
#define _HYBRID_ENGINE_FILTER_HPP_

#include "hybrid_chunk.hpp"

#include <array>

namespace hybrid::engine
{
inline void filter_moments_once(HybridChunk::DataContainer& data)
{
  constexpr std::array<nix::float64, 3> coefficient = {0.25, 0.5, 0.25};
  const bool active_z = !data.particles.empty() && data.particles[0]->has_zdim;
  const bool active_y = !data.particles.empty() && data.particles[0]->has_ydim;
  const bool active_x = !data.particles.empty() && data.particles[0]->has_xdim;

  for (int species = 0; species < data.num_species; ++species) {
    for (int component = 0; component < num_moment_components; ++component) {
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            nix::float64 value = 0;
            for (int dx = active_x ? -1 : 0; dx <= (active_x ? 1 : 0); ++dx) {
              for (int dy = active_y ? -1 : 0; dy <= (active_y ? 1 : 0); ++dy) {
                for (int dz = active_z ? -1 : 0; dz <= (active_z ? 1 : 0); ++dz) {
                  const nix::float64 weight_z = active_z ? coefficient[dz + 1] : 1;
                  const nix::float64 weight_y = active_y ? coefficient[dy + 1] : 1;
                  const nix::float64 weight_x = active_x ? coefficient[dx + 1] : 1;
                  value += weight_z * weight_y * weight_x *
                           data.moment_kinetic(iz + dz, iy + dy, ix + dx, species, component);
                }
              }
            }
            data.filter_scratch(iz, iy, ix) = value;
          }
        }
      }
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            data.moment_kinetic(iz, iy, ix, species, component) = data.filter_scratch(iz, iy, ix);
          }
        }
      }
    }
  }
}

inline void filter_electric_once(HybridChunk::DataContainer& data)
{
  constexpr std::array<nix::float64, 3> coefficient = {0.25, 0.5, 0.25};
  const bool active_z = !data.particles.empty() && data.particles[0]->has_zdim;
  const bool active_y = !data.particles.empty() && data.particles[0]->has_ydim;
  const bool active_x = !data.particles.empty() && data.particles[0]->has_xdim;

  for (int component = field_component::electric_x; component <= field_component::electric_z;
       ++component) {
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          nix::float64 value = 0;
          for (int dz = active_z ? -1 : 0; dz <= (active_z ? 1 : 0); ++dz) {
            for (int dy = active_y ? -1 : 0; dy <= (active_y ? 1 : 0); ++dy) {
              for (int dx = active_x ? -1 : 0; dx <= (active_x ? 1 : 0); ++dx) {
                const nix::float64 weight_z = active_z ? coefficient[dz + 1] : 1;
                const nix::float64 weight_y = active_y ? coefficient[dy + 1] : 1;
                const nix::float64 weight_x = active_x ? coefficient[dx + 1] : 1;
                value += weight_z * weight_y * weight_x *
                         data.work_field_cell(iz + dz, iy + dy, ix + dx, component);
              }
            }
          }
          data.filter_scratch(iz, iy, ix) = value;
        }
      }
    }
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          data.work_field_cell(iz, iy, ix, component) = data.filter_scratch(iz, iy, ix);
        }
      }
    }
  }
}
} // namespace hybrid::engine

#endif
