// -*- C++ -*-
#include "engine/mc2.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;

hybrid::engine::FluidParameters parameters()
{
  return {20.0, 5.0 / 3.0, -2.0, 1.0, 0.7};
}

hybrid::HybridChunk make_mc2_chunk()
{
  const nix::Dims3D   dims{4, 4, 4};
  const nix::Bool3D   has_dim{true, true, true};
  const int           offset[3] = {0, 0, 0};
  const int           global[3] = {4, 4, 4};
  nix::json           config    = {{"Ns", 1},
                                   {"cc", 20.0},
                                   {"gamma", 5.0 / 3.0},
                                   {"delh", 1.0},
                                   {"option", nix::json::object()}};
  hybrid::HybridChunk chunk(dims, has_dim);
  chunk.set_global_context(offset, global);
  chunk.setup(config);
  return chunk;
}
} // namespace

TEST_CASE("MC2 reconstruction matches legacy limiter")
{
  auto monotone = hybrid::engine::mc2_reconstruct(1.0, 2.0, 4.0);
  REQUIRE(monotone.left == Catch::Approx(2.75));
  REQUIRE(monotone.right == Catch::Approx(1.25));

  auto extremum = hybrid::engine::mc2_reconstruct(1.0, 2.0, 1.5);
  REQUIRE(extremum.left == Catch::Approx(2.0));
  REQUIRE(extremum.right == Catch::Approx(2.0));

  auto decreasing = hybrid::engine::mc2_reconstruct(4.0, 2.0, 1.0);
  REQUIRE(decreasing.left == Catch::Approx(1.25));
  REQUIRE(decreasing.right == Catch::Approx(2.75));
}

TEST_CASE("HLL fluid flux matches legacy weighted flux and state jump")
{
  const hybrid::engine::FluidState  left_fluid  = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                   0.8, -0.1, 0.5,  -0.3, 0.6};
  const hybrid::engine::FieldState  left_field  = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const hybrid::engine::FluidState  right_fluid = {1.1, 0.1, 0.25,  -0.15, 0.82,
                                                   0.9, 0.2, -0.35, 0.45,  0.5};
  const hybrid::engine::FieldState  right_field = {-0.1, 0.4, -0.2, 0.8, 0.6, -0.5};
  const hybrid::engine::VectorState background  = {0.2, -0.1, 0.3};

  const auto flux =
      hybrid::engine::hll_fluid_flux(1, left_fluid, left_field, right_fluid, right_field,
                                     background, 1.7, 1.2, 0.05, parameters());
  const hybrid::engine::ConservedState expected_flux = {0.0038620689655172423,
                                                        -0.0028460173548965216, 0.09291378432427518,
                                                        -0.00876579638840072, 0.020571875999980508};
  const hybrid::engine::FieldState expected_field    = {0.07586206896551725,  -0.010344827586206907,
                                                        -0.02413793103448275, 0.9758620689655173,
                                                        -0.16206896551724137, 0.02758620689655175};
  for (int component = 0; component < hybrid::num_conserved_components; ++component) {
    REQUIRE(flux.flux[component] == Catch::Approx(expected_flux[component]).epsilon(tolerance));
  }
  for (int component = 0; component < hybrid::num_field_components; ++component) {
    REQUIRE(flux.field[component] == Catch::Approx(expected_field[component]).epsilon(tolerance));
  }
  REQUIRE_THROWS_AS(hybrid::engine::hll_fluid_flux(3, left_fluid, left_field, right_fluid,
                                                   right_field, background, 1.7, 1.2, 0.05,
                                                   parameters()),
                    std::out_of_range);
  REQUIRE_THROWS_AS(hybrid::engine::hll_fluid_flux(0, left_fluid, left_field, right_fluid,
                                                   right_field, background, 0, 0, 0.05,
                                                   parameters()),
                    std::invalid_argument);
}

TEST_CASE("HLL edge electric formulas preserve legacy signs")
{
  REQUIRE(hybrid::engine::hll_edge_electric_positive(0.6, -0.2, -0.3, 1.4, 1.7, 1.2, 20.0) ==
          Catch::Approx(0.19427586206896552).epsilon(tolerance));
  REQUIRE(hybrid::engine::hll_edge_electric_negative(0.7, -0.4, -0.8, 0.9, 1.7, 1.2, 20.0) ==
          Catch::Approx(0.06262068965517241).epsilon(tolerance));
  REQUIRE_THROWS_AS(hybrid::engine::hll_edge_electric_positive(0.6, -0.2, -0.3, 1.4, 0, 0, 20.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(hybrid::engine::hll_edge_electric_negative(0.7, -0.4, -0.8, 0.9, 1.7, 1.2, 0),
                    std::invalid_argument);
}

TEST_CASE("nonuniform MC2 keeps directional face states and assembles edge electric field")
{
  auto chunk = make_mc2_chunk();
  auto data  = chunk.get_internal_data();
  for (int iz = 0; iz < static_cast<int>(data.work_fluid.shape()[0]); ++iz) {
    for (int iy = 0; iy < static_cast<int>(data.work_fluid.shape()[1]); ++iy) {
      for (int ix = 0; ix < static_cast<int>(data.work_fluid.shape()[2]); ++ix) {
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::electron_density)    = 1;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_x) = 0.01 * ix;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_y) = 0.02 * iy;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_z) = 0.03 * iz;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::electron_pressure)   = 1;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::ion_density)         = 1;
        data.work_fluid(iz, iy, ix, hybrid::fluid_component::ion_pressure)        = 1;
        for (int component = 0; component < hybrid::num_field_components; ++component) {
          const double value = component + 1 + 0.1 * ix + 0.2 * iy + 0.3 * iz;
          data.work_field_cell(iz, iy, ix, component)      = value;
          data.work_field_staggered(iz, iy, ix, component) = value + 0.5;
        }
        for (int direction = 0; direction < hybrid::num_phase_directions; ++direction) {
          data.phase_face(iz, iy, ix, direction, 0) = 2;
          data.phase_face(iz, iy, ix, direction, 1) = 1;
        }
      }
    }
  }

  hybrid::engine::compute_mc2_face_fluxes(data, 0.01, parameters());
  hybrid::engine::compute_mc2_edge_electric(data);

  const int iz = data.Lbz;
  const int iy = data.Lby;
  const int ix = data.Lbx;
  REQUIRE(data.solver_field_x(iz, iy, ix, hybrid::field_component::electric_x) == 0);
  REQUIRE(data.solver_field_y(iz, iy, ix, hybrid::field_component::electric_y) == 0);
  REQUIRE(data.solver_field_z(iz, iy, ix, hybrid::field_component::electric_z) == 0);
  REQUIRE(data.solver_field_x(iz, iy, ix, hybrid::field_component::electric_y) != 0);
  REQUIRE(data.solver_field_y(iz, iy, ix, hybrid::field_component::electric_x) != 0);
  REQUIRE(data.solver_field_z(iz, iy, ix, hybrid::field_component::electric_x) != 0);
  REQUIRE(data.solver_field_x(iz, iy, ix, hybrid::field_component::magnetic_x) ==
          data.work_field_staggered(iz, iy, ix, hybrid::field_component::magnetic_x));
  for (int component = 0; component < hybrid::num_vector_components; ++component) {
    REQUIRE(std::isfinite(data.field_flux(iz, iy, ix, component)));
    REQUIRE(data.field_flux(iz, iy, ix, component) != 0);
  }
}
