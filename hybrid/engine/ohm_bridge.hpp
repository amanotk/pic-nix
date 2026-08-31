// -*- C++ -*-
#ifndef _HYBRID_ENGINE_OHM_BRIDGE_HPP_
#define _HYBRID_ENGINE_OHM_BRIDGE_HPP_

#include "engine/fluid.hpp"
#include "engine/ohm_source.hpp"
#include "engine/ssor2.hpp"

#include <stdexcept>
#include <vector>

namespace hybrid::engine
{
inline std::array<nix::float64, num_moment_components>
fluid_to_moment(const FluidState& fluid, nix::float64 ion_charge_to_mass,
                nix::float64 electron_charge_to_mass)
{
  const auto&        e                = &fluid[0];
  const auto&        i                = &fluid[5];
  const nix::float64 ion_linear       = nix::math::pi4 * ion_charge_to_mass;
  const nix::float64 electron_linear  = nix::math::pi4 * electron_charge_to_mass;
  const nix::float64 ion_squared      = ion_linear * ion_charge_to_mass;
  const nix::float64 electron_squared = electron_linear * electron_charge_to_mass;
  return {electron_squared * e[0] + ion_squared * i[0],
          electron_squared * e[0] * e[1] + ion_squared * i[0] * i[1],
          electron_squared * e[0] * e[2] + ion_squared * i[0] * i[2],
          electron_squared * e[0] * e[3] + ion_squared * i[0] * i[3],
          electron_linear * (e[0] * e[1] * e[1] + e[4]) + ion_linear * (i[0] * i[1] * i[1] + i[4]),
          electron_linear * (e[0] * e[2] * e[2] + e[4]) + ion_linear * (i[0] * i[2] * i[2] + i[4]),
          electron_linear * (e[0] * e[3] * e[3] + e[4]) + ion_linear * (i[0] * i[3] * i[3] + i[4]),
          electron_linear * e[0] * e[1] * e[2] + ion_linear * i[0] * i[1] * i[2],
          electron_linear * e[0] * e[1] * e[3] + ion_linear * i[0] * i[1] * i[3],
          electron_linear * e[0] * e[2] * e[3] + ion_linear * i[0] * i[2] * i[3]};
}

inline void accumulate_fluid_moment(const FluidState& fluid, nix::float64 ion_charge_to_mass,
                                    nix::float64                electron_charge_to_mass,
                                    nix::Array4D<nix::float64>& ohm_moment, int iz, int iy, int ix)
{
  const auto m = fluid_to_moment(fluid, ion_charge_to_mass, electron_charge_to_mass);
  for (int component = 0; component < num_moment_components; ++component) {
    ohm_moment(iz, iy, ix, component) += m[component];
  }
}

inline void accumulate_kinetic_moments(const nix::Array5D<nix::float64>& moment_kinetic,
                                       nix::Array4D<nix::float64>& ohm_moment, int iz, int iy,
                                       int ix, const std::vector<nix::float64>& charge_to_mass)
{
  if (moment_kinetic.shape()[3] != charge_to_mass.size()) {
    throw std::invalid_argument("Hybrid kinetic Ohm moment species count mismatch");
  }
  for (int species = 0; species < static_cast<int>(charge_to_mass.size()); ++species) {
    const nix::float64 linear  = nix::math::pi4 * charge_to_mass[species];
    const nix::float64 squared = linear * charge_to_mass[species];
    for (int component = 0; component < num_moment_components; ++component) {
      const nix::float64 weight = component <= moment_component::momentum_z ? squared : linear;
      ohm_moment(iz, iy, ix, component) += weight * moment_kinetic(iz, iy, ix, species, component);
    }
  }
}

inline OhmSolveStats solve_ssor2_electric(nix::Array4D<nix::float64>& field_cell,
                                          nix::Array4D<nix::float64>& ohm_source, int Lbx, int Ubx,
                                          int Lby, int Uby, int Lbz, int Ubz,
                                          nix::float64 light_speed, nix::float64 spacing_x,
                                          nix::float64 spacing_y, nix::float64 spacing_z,
                                          int max_iterations, nix::float64 tolerance)
{
  Ssor2Workspace workspace = {field_cell, ohm_source, Lbx, Ubx, Lby, Uby, Lbz, Ubz};
  const auto     coeff = compute_ssor2_coefficients(light_speed, spacing_x, spacing_y, spacing_z);
  const Ssor2Config config{max_iterations, tolerance};
  return solve_ssor2(workspace, coeff, config);
}
} // namespace hybrid::engine

#endif
