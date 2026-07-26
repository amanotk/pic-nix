// -*- C++ -*-
#ifndef _HYBRID_ENGINE_FLUID_HPP_
#define _HYBRID_ENGINE_FLUID_HPP_

#include "hybrid.hpp"
#include "nix/nix.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace hybrid::engine
{
using FluidState     = std::array<nix::float64, num_fluid_components>;
using FieldState     = std::array<nix::float64, num_field_components>;
using ConservedState = std::array<nix::float64, num_conserved_components>;
using VectorState    = std::array<nix::float64, num_vector_components>;
using CurrentState   = std::array<nix::float64, num_current_components>;

struct FluidParameters {
  nix::float64 light_speed;
  nix::float64 adiabatic_index;
  nix::float64 electron_charge_to_mass;
  nix::float64 ion_charge_to_mass;
  nix::float64 electron_entropy;
};

inline ConservedState conservative(const FluidState& fluid, const FieldState& field,
                                   const FluidParameters& parameters)
{
  constexpr nix::float64 inverse_four_pi = 1.0 / nix::math::pi4;
  const auto*            electron        = &fluid[0];
  const auto*            ion             = &fluid[5];
  ConservedState         result          = {};
  result[0]                              = ion[0] + electron[0];
  for (int component = 0; component < 3; ++component) {
    result[component + 1] = ion[0] * ion[component + 1] + electron[0] * electron[component + 1];
  }
  const nix::float64 ion_kinetic =
      0.5 * ion[0] * (ion[1] * ion[1] + ion[2] * ion[2] + ion[3] * ion[3]);
  const nix::float64 electron_kinetic =
      0.5 * electron[0] *
      (electron[1] * electron[1] + electron[2] * electron[2] + electron[3] * electron[3]);
  const nix::float64 magnetic =
      0.5 * (field[3] * field[3] + field[4] * field[4] + field[5] * field[5]) * inverse_four_pi;
  result[4] = ion_kinetic + electron_kinetic +
              (ion[4] + electron[4]) / (parameters.adiabatic_index - 1.0) + magnetic;
  return result;
}

inline FluidState primitive(const ConservedState& conserved, const FieldState& field,
                            const VectorState& curl_b, const CurrentState& current,
                            const FluidParameters& parameters)
{
  constexpr nix::float64 epsilon         = 1.0e-32;
  constexpr nix::float64 inverse_four_pi = 1.0 / nix::math::pi4;
  const nix::float64     signal =
      parameters.ion_charge_to_mass / (parameters.ion_charge_to_mass + epsilon);
  const nix::float64 inverse_charge_difference =
      1.0 / (parameters.ion_charge_to_mass - parameters.electron_charge_to_mass);
  FluidState result   = {};
  auto*      electron = &result[0];
  auto*      ion      = &result[5];
  electron[0] =
      (current[0] + parameters.ion_charge_to_mass * conserved[0]) * inverse_charge_difference;
  ion[0]                                      = (conserved[0] - electron[0]) * signal;
  const nix::float64 inverse_electron_density = 1.0 / (electron[0] + epsilon);
  const nix::float64 inverse_ion_density      = (1.0 / (ion[0] + epsilon)) * signal;
  for (int component = 0; component < 3; ++component) {
    electron[component + 1] = (current[component + 1] - curl_b[component] +
                               parameters.ion_charge_to_mass * conserved[component + 1]) *
                              inverse_charge_difference * inverse_electron_density;
    ion[component + 1] =
        (conserved[component + 1] - electron[0] * electron[component + 1]) * inverse_ion_density;
  }
  const nix::float64 ion_kinetic =
      0.5 * ion[0] * (ion[1] * ion[1] + ion[2] * ion[2] + ion[3] * ion[3]);
  const nix::float64 electron_kinetic =
      0.5 * electron[0] *
      (electron[1] * electron[1] + electron[2] * electron[2] + electron[3] * electron[3]);
  const nix::float64 magnetic =
      0.5 * (field[3] * field[3] + field[4] * field[4] + field[5] * field[5]) * inverse_four_pi;
  const nix::float64 pressure = (conserved[4] - ion_kinetic - electron_kinetic - magnetic) *
                                (parameters.adiabatic_index - 1.0);
  electron[4] = parameters.electron_entropy * std::pow(electron[0], parameters.adiabatic_index);
  ion[4]      = pressure - electron[4];
  return result;
}

inline ConservedState fluid_rhs(nix::float64 time_step, const FieldState& field,
                                const CurrentState& current, const VectorState& background,
                                const FluidParameters& parameters)
{
  const nix::float64 charge                 = -current[0] * time_step;
  const nix::float64 jx                     = -current[1] * time_step;
  const nix::float64 jy                     = -current[2] * time_step;
  const nix::float64 jz                     = -current[3] * time_step;
  const nix::float64 bx                     = field[3] + background[0];
  const nix::float64 by                     = field[4] + background[1];
  const nix::float64 bz                     = field[5] + background[2];
  const nix::float64 reciprocal_light_speed = 1.0 / parameters.light_speed;
  return {0, charge * field[0] + (jy * bz - jz * by) * reciprocal_light_speed,
          charge * field[1] + (jz * bx - jx * bz) * reciprocal_light_speed,
          charge * field[2] + (jx * by - jy * bx) * reciprocal_light_speed,
          jx * field[0] + jy * field[1] + jz * field[2]};
}

inline ConservedState physical_flux(int direction, const FluidState& fluid, const FieldState& field,
                                    const VectorState&     background,
                                    const FluidParameters& parameters)
{
  if (direction < 0 || direction >= num_vector_components) {
    throw std::out_of_range("Hybrid fluid flux direction must be 0, 1, or 2");
  }
  constexpr nix::float64 inverse_four_pi    = 1.0 / nix::math::pi4;
  const auto*            electron           = &fluid[0];
  const auto*            ion                = &fluid[5];
  const int              normal_velocity    = direction + 1;
  const nix::float64     ion_mass_flux      = ion[0] * ion[normal_velocity];
  const nix::float64     electron_mass_flux = electron[0] * electron[normal_velocity];
  const nix::float64     ion_specific_kinetic =
      0.5 * (ion[1] * ion[1] + ion[2] * ion[2] + ion[3] * ion[3]);
  const nix::float64 electron_specific_kinetic =
      0.5 * (electron[1] * electron[1] + electron[2] * electron[2] + electron[3] * electron[3]);
  VectorState        magnetic = {field[3] + background[0], field[4] + background[1],
                                 field[5] + background[2]};
  const nix::float64 background_energy =
      0.5 *
      (background[0] * background[0] + background[1] * background[1] +
       background[2] * background[2]) *
      inverse_four_pi;
  const nix::float64 magnetic_energy =
      0.5 * (magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2]) *
      inverse_four_pi;
  const nix::float64 total_pressure = magnetic_energy - background_energy + ion[4] + electron[4];
  const VectorState  poynting       = {
             field[1] * field[5] - field[2] * field[4],
             field[2] * field[3] - field[0] * field[5],
             field[0] * field[4] - field[1] * field[3],
  };

  ConservedState result = {};
  result[0]             = ion_mass_flux + electron_mass_flux;
  for (int component = 0; component < 3; ++component) {
    const nix::float64 pressure = component == direction ? total_pressure : 0;
    const nix::float64 stress   = (magnetic[direction] * magnetic[component] -
                                 background[direction] * background[component]) *
                                inverse_four_pi;
    result[component + 1] = ion_mass_flux * ion[component + 1] +
                            electron_mass_flux * electron[component + 1] + pressure - stress;
  }
  result[4] = ion_mass_flux * ion_specific_kinetic +
              electron_mass_flux * electron_specific_kinetic +
              parameters.adiabatic_index / (parameters.adiabatic_index - 1.0) *
                  (ion[4] * ion[normal_velocity] + electron[4] * electron[normal_velocity]) +
              parameters.light_speed * poynting[direction] * inverse_four_pi;
  return result;
}
} // namespace hybrid::engine

#endif
