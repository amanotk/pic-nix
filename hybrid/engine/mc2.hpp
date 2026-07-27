// -*- C++ -*-
#ifndef _HYBRID_ENGINE_MC2_HPP_
#define _HYBRID_ENGINE_MC2_HPP_

#include "engine/fluid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace hybrid::engine
{
struct Mc2Reconstruction {
  nix::float64 left;
  nix::float64 right;
};

struct HllFluidFlux {
  ConservedState flux;
  FieldState     field;
};

inline nix::float64 limiter_sign(nix::float64 value)
{
  return std::copysign(1.0, value);
}

inline nix::float64 mc2_limiter(nix::float64 a, nix::float64 b)
{
  return 0.5 * (limiter_sign(a) + limiter_sign(b)) *
         std::min({2 * std::abs(a), 2 * std::abs(b), 0.5 * std::abs(a + b)});
}

inline Mc2Reconstruction mc2_reconstruct(nix::float64 minus, nix::float64 center, nix::float64 plus)
{
  const nix::float64 slope = mc2_limiter(center - minus, plus - center);
  return {center + 0.5 * slope, center - 0.5 * slope};
}

inline HllFluidFlux hll_fluid_flux(int direction, const FluidState& left_fluid,
                                   const FieldState& left_field, const FluidState& right_fluid,
                                   const FieldState& right_field, const VectorState& background,
                                   nix::float64 phase_max, nix::float64 phase_min,
                                   nix::float64 time_step, const FluidParameters& parameters)
{
  if (direction < 0 || direction >= num_phase_directions) {
    throw std::out_of_range("Hybrid HLL fluid direction must be 0, 1, or 2");
  }
  const nix::float64 reciprocal_speed = 1.0 / (phase_max + phase_min);
  const auto left_flux = physical_flux(direction, left_fluid, left_field, background, parameters);
  const auto right_flux =
      physical_flux(direction, right_fluid, right_field, background, parameters);
  const auto left_conserved  = conservative(left_fluid, left_field, parameters);
  const auto right_conserved = conservative(right_fluid, right_field, parameters);

  HllFluidFlux result = {};
  for (int component = 0; component < num_conserved_components; ++component) {
    result.flux[component] =
        (phase_max * left_flux[component] + phase_min * right_flux[component] -
         phase_max * phase_min * (right_conserved[component] - left_conserved[component])) *
        reciprocal_speed * time_step;
  }
  for (int component = 0; component < num_field_components; ++component) {
    result.field[component] =
        (phase_max * left_field[component] + phase_min * right_field[component]) * reciprocal_speed;
  }
  return result;
}

inline nix::float64 hll_edge_electric_positive(nix::float64 left_electric,
                                               nix::float64 right_electric,
                                               nix::float64 left_magnetic,
                                               nix::float64 right_magnetic, nix::float64 phase_max,
                                               nix::float64 phase_min, nix::float64 light_speed)
{
  const nix::float64 reciprocal_speed       = 1.0 / (phase_max + phase_min);
  const nix::float64 reciprocal_light_speed = 1.0 / light_speed;
  return (0.5 * (phase_max * left_electric + phase_min * right_electric) +
          reciprocal_light_speed * phase_max * phase_min * (right_magnetic - left_magnetic)) *
         reciprocal_speed;
}

inline nix::float64 hll_edge_electric_negative(nix::float64 left_electric,
                                               nix::float64 right_electric,
                                               nix::float64 left_magnetic,
                                               nix::float64 right_magnetic, nix::float64 phase_max,
                                               nix::float64 phase_min, nix::float64 light_speed)
{
  const nix::float64 reciprocal_speed       = 1.0 / (phase_max + phase_min);
  const nix::float64 reciprocal_light_speed = 1.0 / light_speed;
  return (0.5 * (phase_max * left_electric + phase_min * right_electric) -
          reciprocal_light_speed * phase_max * phase_min * (right_magnetic - left_magnetic)) *
         reciprocal_speed;
}
} // namespace hybrid::engine

#endif
