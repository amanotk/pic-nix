// -*- C++ -*-
#ifndef _HYBRID_ENGINE_PHASESPEED_HPP_
#define _HYBRID_ENGINE_PHASESPEED_HPP_

#include "engine/fluid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace hybrid::engine
{
using PhaseState = std::array<nix::float64, num_phase_directions * num_phase_branches>;

struct PhaseSpeedParameters {
  nix::float64 light_speed;
  nix::float64 adiabatic_index;
  nix::float64 ion_charge_to_mass;
  nix::float64 electron_charge_to_mass;
  nix::float64 spacing_x;
  nix::float64 spacing_y;
  nix::float64 spacing_z;
  int          max_iterations = 20;
  nix::float64 tolerance      = 1.0e-4;
};

inline std::array<nix::float64, num_phase_directions>
nyquist_wavenumber_squared(const PhaseSpeedParameters& parameters)
{
  std::array<nix::float64, num_phase_directions> result = {nix::math::pi / parameters.spacing_x,
                                                           nix::math::pi / parameters.spacing_y,
                                                           nix::math::pi / parameters.spacing_z};
  for (auto& value : result) {
    value *= value;
  }
  return result;
}

inline nix::float64 phase_cubic(const nix::float64 x, const std::array<nix::float64, 4>& c)
{
  return x * (x * (x * c[0] + c[1]) + c[2]) + c[3];
}

inline nix::float64 phase_cubic_first_derivative(const nix::float64                 x,
                                                 const std::array<nix::float64, 4>& c)
{
  return x * (3 * x * c[0] + 2 * c[1]) + c[2];
}

inline nix::float64 phase_cubic_second_derivative_legacy(const nix::float64                 x,
                                                         const std::array<nix::float64, 4>& c)
{
  return x * 6 * c[0] + c[1];
}

inline nix::float64 solve_phase_cubic(const nix::float64 k2, const nix::float64 ct2,
                                      const nix::float64 beta,
                                      const nix::float64 electron_ion_ratio, const nix::float64 x0,
                                      const PhaseSpeedParameters& parameters)
{
  const nix::float64                p = 1 + electron_ion_ratio * k2;
  const std::array<nix::float64, 4> c = {
      p * p,
      -(1 + (1 + k2) * ct2 + electron_ion_ratio * (1 + (electron_ion_ratio - 1) * ct2) * k2 +
        beta * p * p),
      ct2 * (1 + 2 * beta + (1 + electron_ion_ratio * electron_ion_ratio) * beta * k2),
      -beta * ct2 * ct2};

  const nix::float64 maximum = (-c[1] + std::sqrt(c[1] * c[1] - 3 * c[0] * c[2])) / (3 * c[0]);
  nix::float64       x       = std::max(maximum * 1.001, x0);
  for (int iteration = 1; iteration <= parameters.max_iterations; ++iteration) {
    const nix::float64 ff = phase_cubic(x, c);
    const nix::float64 gg = phase_cubic_first_derivative(x, c);
    const nix::float64 hh = phase_cubic_second_derivative_legacy(x, c);
    const nix::float64 dx = -2 * ff * gg / (2 * gg * gg - ff * hh);
    x += dx;
    if (std::abs(dx / x) < parameters.tolerance) {
      break;
    }
  }
  return x;
}

inline PhaseState
rest_frame_phase_speed(const FluidState& fluid, const FieldState& field,
                       const std::array<nix::float64, num_vector_components>& background,
                       const PhaseSpeedParameters&                            parameters)
{
  const auto         kn  = nyquist_wavenumber_squared(parameters);
  const nix::float64 cc  = parameters.light_speed * parameters.light_speed;
  const nix::float64 wpi = fluid[fluid_component::ion_density] * parameters.ion_charge_to_mass *
                           parameters.ion_charge_to_mass * nix::math::pi4;
  const nix::float64 wpe = fluid[fluid_component::electron_density] *
                           parameters.electron_charge_to_mass * parameters.electron_charge_to_mass *
                           nix::math::pi4;
  const nix::float64 ro =
      fluid[fluid_component::electron_density] + fluid[fluid_component::ion_density];
  const nix::float64 pr =
      fluid[fluid_component::electron_pressure] + fluid[fluid_component::ion_pressure];
  const nix::float64                                    rki      = cc / wpi;
  const nix::float64                                    mei      = wpi / wpe;
  const std::array<nix::float64, num_vector_components> magnetic = {
      field[field_component::magnetic_x] + background[0],
      field[field_component::magnetic_y] + background[1],
      field[field_component::magnetic_z] + background[2]};

  const nix::float64 b2 =
      magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2] + 1.0e-30;
  const nix::float64 va2  = b2 * (1.0 / nix::math::pi4) / ro;
  const nix::float64 vs2  = parameters.adiabatic_index * pr / ro;
  const nix::float64 beta = vs2 / va2;

  PhaseState result = {};
  for (int direction = 0; direction < num_phase_directions; ++direction) {
    const nix::float64 ct2 = magnetic[direction] * magnetic[direction] / b2;
    const nix::float64 k2  = kn[direction] * rki;
    const nix::float64 root =
        solve_phase_cubic(k2, ct2, beta, mei,
                          result[3 * direction + 2] * result[3 * direction + 2] / va2, parameters);
    result[3 * direction + 2] = std::sqrt(root * va2);
  }
  return result;
}

inline PhaseState
default_phase_speed(const FluidState& fluid, const FieldState& field,
                    const std::array<nix::float64, num_vector_components>& background,
                    const PhaseSpeedParameters&                            parameters)
{
  PhaseState         result = rest_frame_phase_speed(fluid, field, background, parameters);
  const auto         kn     = nyquist_wavenumber_squared(parameters);
  const nix::float64 cc     = parameters.light_speed * parameters.light_speed;
  const nix::float64 wpe    = fluid[fluid_component::electron_density] *
                           parameters.electron_charge_to_mass * parameters.electron_charge_to_mass *
                           nix::math::pi4;
  const nix::float64 rke = cc / wpe;
  for (int direction = 0; direction < num_phase_directions; ++direction) {
    const int          index = 3 * direction;
    const nix::float64 phase = result[index + 2];
    const nix::float64 ve    = fluid[fluid_component::electron_velocity_x + direction];
    const nix::float64 vi    = fluid[fluid_component::ion_velocity_x + direction];
    const nix::float64 psmax = std::max({ve + phase, vi + phase, 0.0});
    const nix::float64 psmin = -std::min({ve - phase, vi - phase, 0.0});
    result[index]            = psmax;
    result[index + 1]        = psmin;
    if (kn[direction] * rke > 1.0) {
      const nix::float64 symmetric = std::max(psmax, psmin);
      result[index]                = symmetric;
      result[index + 1]            = symmetric;
    }
  }
  return result;
}

inline PhaseState phase_cell_to_face(const PhaseState& left, const PhaseState& right, int direction)
{
  if (direction < 0 || direction >= num_phase_directions) {
    throw std::out_of_range("Hybrid phase-speed direction must be 0, 1, or 2");
  }
  PhaseState result = {};
  const int  index  = 3 * direction;
  result[index]     = std::max(left[index], right[index]);
  result[index + 1] = std::max(left[index + 1], right[index + 1]);
  return result;
}
} // namespace hybrid::engine

#endif
