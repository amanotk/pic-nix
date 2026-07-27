// -*- C++ -*-
#ifndef _HYBRID_ENGINE_MC2_HPP_
#define _HYBRID_ENGINE_MC2_HPP_

#include "engine/fluid.hpp"
#include "hybrid_chunk.hpp"

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

inline nix::float64 reciprocal_hll_speed(nix::float64 phase_max, nix::float64 phase_min)
{
  const nix::float64 speed_sum = phase_max + phase_min;
  if (!std::isfinite(phase_max) || !std::isfinite(phase_min) || phase_max < 0 || phase_min < 0 ||
      speed_sum <= 0) {
    throw std::invalid_argument("Hybrid HLL phase speeds must be finite, nonnegative, and nonzero");
  }
  return 1.0 / speed_sum;
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
  const nix::float64 reciprocal_speed = reciprocal_hll_speed(phase_max, phase_min);
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
  if (!std::isfinite(light_speed) || light_speed == 0) {
    throw std::invalid_argument("Hybrid edge-electric HLL requires finite nonzero light speed");
  }
  const nix::float64 reciprocal_speed       = reciprocal_hll_speed(phase_max, phase_min);
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
  if (!std::isfinite(light_speed) || light_speed == 0) {
    throw std::invalid_argument("Hybrid edge-electric HLL requires finite nonzero light speed");
  }
  const nix::float64 reciprocal_speed       = reciprocal_hll_speed(phase_max, phase_min);
  const nix::float64 reciprocal_light_speed = 1.0 / light_speed;
  return (0.5 * (phase_max * left_electric + phase_min * right_electric) -
          reciprocal_light_speed * phase_max * phase_min * (right_magnetic - left_magnetic)) *
         reciprocal_speed;
}

inline std::array<nix::float64, 2> mc2_face_pair(const nix::Array4D<nix::float64>& array,
                                                 int direction, int iz, int iy, int ix,
                                                 int component)
{
  const auto value = [&](int offset) {
    return array(iz + (direction == 2 ? offset : 0), iy + (direction == 1 ? offset : 0),
                 ix + (direction == 0 ? offset : 0), component);
  };
  return {mc2_reconstruct(value(-1), value(0), value(1)).left,
          mc2_reconstruct(value(0), value(1), value(2)).right};
}

inline void compute_mc2_face_fluxes(HybridChunk::DataContainer& data, nix::float64 time_step,
                                    const FluidParameters& parameters)
{
  data.fluid_flux.fill(0);
  data.solver_field_x.fill(0);
  data.solver_field_y.fill(0);
  data.solver_field_z.fill(0);

  for (int direction = 0; direction < num_phase_directions; ++direction) {
    auto&     face_field = direction == 0   ? data.solver_field_x
                           : direction == 1 ? data.solver_field_y
                                            : data.solver_field_z;
    auto&     background = direction == 0   ? data.background_x_face
                           : direction == 1 ? data.background_y_face
                                            : data.background_z_face;
    const int z_begin    = direction == 2 ? data.Lbz - 1 : 0;
    const int z_end      = direction == 2 ? data.Ubz : static_cast<int>(face_field.shape()[0]) - 1;
    const int y_begin    = direction == 1 ? data.Lby - 1 : 0;
    const int y_end      = direction == 1 ? data.Uby : static_cast<int>(face_field.shape()[1]) - 1;
    const int x_begin    = direction == 0 ? data.Lbx - 1 : 0;
    const int x_end      = direction == 0 ? data.Ubx : static_cast<int>(face_field.shape()[2]) - 1;

    for (int iz = z_begin; iz <= z_end; ++iz) {
      for (int iy = y_begin; iy <= y_end; ++iy) {
        for (int ix = x_begin; ix <= x_end; ++ix) {
          FluidState  left_fluid = {}, right_fluid = {};
          FieldState  left_field = {}, right_field = {};
          VectorState face_background = {};
          for (int component = 0; component < num_fluid_components; ++component) {
            const auto reconstructed =
                mc2_face_pair(data.work_fluid, direction, iz, iy, ix, component);
            left_fluid[component]  = reconstructed[0];
            right_fluid[component] = reconstructed[1];
          }
          for (int component = 0; component < num_field_components; ++component) {
            if (component == field_component::electric_x + direction) {
              continue;
            }
            if (component == field_component::magnetic_x + direction) {
              const auto normal      = data.work_field_staggered(iz, iy, ix, component);
              left_field[component]  = normal;
              right_field[component] = normal;
            } else {
              const auto reconstructed =
                  mc2_face_pair(data.work_field_cell, direction, iz, iy, ix, component);
              left_field[component]  = reconstructed[0];
              right_field[component] = reconstructed[1];
            }
          }
          for (int component = 0; component < num_vector_components; ++component) {
            face_background[component] = background(iz, iy, ix, component);
          }
          const auto flux =
              hll_fluid_flux(direction, left_fluid, left_field, right_fluid, right_field,
                             face_background, data.phase_face(iz, iy, ix, direction, 0),
                             data.phase_face(iz, iy, ix, direction, 1), time_step, parameters);
          for (int component = 0; component < num_conserved_components; ++component) {
            if (!std::isfinite(flux.flux[component])) {
              throw std::runtime_error("Hybrid MC2 produced non-finite fluid flux");
            }
            data.fluid_flux(iz, iy, ix, direction, component) = flux.flux[component];
          }
          for (int component = 0; component < num_field_components; ++component) {
            if (!std::isfinite(flux.field[component])) {
              throw std::runtime_error("Hybrid MC2 produced non-finite face field");
            }
            face_field(iz, iy, ix, component) = flux.field[component];
          }
        }
      }
    }
  }
}

inline void compute_mc2_edge_electric(HybridChunk::DataContainer& data)
{
  data.field_flux.fill(0);
  const auto phase_average = [&](int direction, int branch, int iz0, int iy0, int ix0, int iz1,
                                 int iy1, int ix1) {
    return 0.5 * (data.phase_face(iz0, iy0, ix0, direction, branch) +
                  data.phase_face(iz1, iy1, ix1, direction, branch));
  };
  const auto positive = [&](const nix::Array4D<nix::float64>& array, int direction, int iz, int iy,
                            int ix, int electric_component, int magnetic_component,
                            nix::float64 phase_max, nix::float64 phase_min) {
    const auto electric = mc2_face_pair(array, direction, iz, iy, ix, electric_component);
    const auto magnetic = mc2_face_pair(array, direction, iz, iy, ix, magnetic_component);
    return hll_edge_electric_positive(electric[0], electric[1], magnetic[0], magnetic[1], phase_max,
                                      phase_min, data.light_speed);
  };
  const auto negative = [&](const nix::Array4D<nix::float64>& array, int direction, int iz, int iy,
                            int ix, int electric_component, int magnetic_component,
                            nix::float64 phase_max, nix::float64 phase_min) {
    const auto electric = mc2_face_pair(array, direction, iz, iy, ix, electric_component);
    const auto magnetic = mc2_face_pair(array, direction, iz, iy, ix, magnetic_component);
    return hll_edge_electric_negative(electric[0], electric[1], magnetic[0], magnetic[1], phase_max,
                                      phase_min, data.light_speed);
  };

  for (int iz = data.Lbz - 1; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby - 1; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx - 1; ix <= data.Ubx; ++ix) {
        auto vmax = phase_average(0, 0, iz, iy, ix, iz, iy + 1, ix);
        auto vmin = phase_average(0, 1, iz, iy, ix, iz, iy + 1, ix);
        data.field_flux(iz, iy, ix, field_component::electric_z) +=
            positive(data.solver_field_y, 0, iz, iy, ix, field_component::electric_z,
                     field_component::magnetic_y, vmax, vmin);
        vmax = phase_average(0, 0, iz, iy, ix, iz + 1, iy, ix);
        vmin = phase_average(0, 1, iz, iy, ix, iz + 1, iy, ix);
        data.field_flux(iz, iy, ix, field_component::electric_y) +=
            negative(data.solver_field_z, 0, iz, iy, ix, field_component::electric_y,
                     field_component::magnetic_z, vmax, vmin);

        vmax = phase_average(1, 0, iz, iy, ix, iz + 1, iy, ix);
        vmin = phase_average(1, 1, iz, iy, ix, iz + 1, iy, ix);
        data.field_flux(iz, iy, ix, field_component::electric_x) +=
            positive(data.solver_field_z, 1, iz, iy, ix, field_component::electric_x,
                     field_component::magnetic_z, vmax, vmin);
        vmax = phase_average(1, 0, iz, iy, ix, iz, iy, ix + 1);
        vmin = phase_average(1, 1, iz, iy, ix, iz, iy, ix + 1);
        data.field_flux(iz, iy, ix, field_component::electric_z) +=
            negative(data.solver_field_x, 1, iz, iy, ix, field_component::electric_z,
                     field_component::magnetic_x, vmax, vmin);

        vmax = phase_average(2, 0, iz, iy, ix, iz, iy, ix + 1);
        vmin = phase_average(2, 1, iz, iy, ix, iz, iy, ix + 1);
        data.field_flux(iz, iy, ix, field_component::electric_y) +=
            positive(data.solver_field_x, 2, iz, iy, ix, field_component::electric_y,
                     field_component::magnetic_x, vmax, vmin);
        vmax = phase_average(2, 0, iz, iy, ix, iz, iy + 1, ix);
        vmin = phase_average(2, 1, iz, iy, ix, iz, iy + 1, ix);
        data.field_flux(iz, iy, ix, field_component::electric_x) +=
            negative(data.solver_field_y, 2, iz, iy, ix, field_component::electric_x,
                     field_component::magnetic_y, vmax, vmin);

        for (int component = 0; component < num_vector_components; ++component) {
          if (!std::isfinite(data.field_flux(iz, iy, ix, component))) {
            throw std::runtime_error("Hybrid MC2 produced non-finite edge electric field");
          }
        }
      }
    }
  }
}
} // namespace hybrid::engine

#endif
