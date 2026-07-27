// -*- C++ -*-
#ifndef _HYBRID_ENGINE_OHM_BRIDGE_HPP_
#define _HYBRID_ENGINE_OHM_BRIDGE_HPP_

#include "engine/fluid.hpp"
#include "engine/ohm_source.hpp"
#include "engine/ssor2.hpp"

namespace hybrid::engine
{
inline std::array<nix::float64, num_moment_components> fluid_to_moment(const FluidState& fluid)
{
  const auto& e = &fluid[0];
  const auto& i = &fluid[5];
  return {e[0] + i[0],
          e[0] * e[1] + i[0] * i[1],
          e[0] * e[2] + i[0] * i[2],
          e[0] * e[3] + i[0] * i[3],
          e[0] * e[1] * e[1] + i[0] * i[1] * i[1] + e[4] + i[4],
          e[0] * e[2] * e[2] + i[0] * i[2] * i[2] + e[4] + i[4],
          e[0] * e[3] * e[3] + i[0] * i[3] * i[3] + e[4] + i[4],
          e[0] * e[1] * e[2] + i[0] * i[1] * i[2],
          e[0] * e[1] * e[3] + i[0] * i[1] * i[3],
          e[0] * e[2] * e[3] + i[0] * i[2] * i[3]};
}

inline void accumulate_fluid_moment(const FluidState& fluid, nix::Array4D<nix::float64>& ohm_moment,
                                    int iz, int iy, int ix)
{
  const auto m = fluid_to_moment(fluid);
  for (int component = 0; component < num_moment_components; ++component) {
    ohm_moment(iz, iy, ix, component) += m[component];
  }
}

inline void accumulate_kinetic_moments(const nix::Array5D<nix::float64>& moment_kinetic,
                                       nix::Array4D<nix::float64>& ohm_moment, int iz, int iy,
                                       int ix, int num_species)
{
  for (int species = 0; species < num_species; ++species) {
    for (int component = 0; component < num_moment_components; ++component) {
      ohm_moment(iz, iy, ix, component) += moment_kinetic(iz, iy, ix, species, component);
    }
  }
}

inline void solve_ssor2_electric(nix::Array4D<nix::float64>& field_cell,
                                 nix::Array4D<nix::float64>& ohm_source,
                                 nix::Array4D<nix::float64>& resistive, int Lbx, int Ubx, int Lby,
                                 int Uby, int Lbz, int Ubz, nix::float64 light_speed,
                                 nix::float64 spacing_x, nix::float64 spacing_y,
                                 nix::float64 spacing_z, int max_iterations, nix::float64 tolerance)
{
  Ssor2Workspace workspace = {field_cell, ohm_source, resistive, Lbx, Ubx, Lby, Uby, Lbz, Ubz};
  const auto     coeff = compute_ssor2_coefficients(light_speed, spacing_x, spacing_y, spacing_z);
  const Ssor2Config config{max_iterations, tolerance};
  solve_ssor2(workspace, coeff, config);
}
} // namespace hybrid::engine

#endif
