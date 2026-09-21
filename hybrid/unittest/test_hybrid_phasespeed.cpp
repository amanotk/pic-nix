// -*- C++ -*-
#include "engine/phasespeed.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;

hybrid::engine::PhaseSpeedParameters parameters()
{
  return {20.0, 5.0 / 3.0, 1.0, -2.0, 0.5, 2.0, 4.0, 20, 1.0e-4};
}
} // namespace

TEST_CASE("phase cubic solver preserves legacy Halley derivative")
{
  const auto root = hybrid::engine::solve_phase_cubic(3.0, 0.4, 0.8, 0.25, 0.0, parameters());
  REQUIRE(root == Catch::Approx(1.342580584291956).epsilon(tolerance));
}

TEST_CASE("default phase speed matches legacy two-fluid formula")
{
  const auto                        params     = parameters();
  const hybrid::engine::FluidState  fluid      = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                  0.8, -0.1, 0.5,  -0.3, 0.6};
  const hybrid::engine::FieldState  field      = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const hybrid::engine::VectorState background = {0.2, -0.1, 0.3};

  const auto phase = hybrid::engine::default_phase_speed(fluid, field, background, params);
  const hybrid::engine::PhaseState expected = {
      1.4360640497253883, 1.4360640497253883, 1.1360640497253882,
      1.638197748183572,  1.638197748183572,  1.138197748183572,
      1.5440791010176116, 1.5440791010176116, 1.1440791010176117};
  for (int component = 0; component < hybrid::num_phase_directions * hybrid::num_phase_branches;
       ++component) {
    REQUIRE(phase[component] == Catch::Approx(expected[component]).epsilon(tolerance));
  }
}

TEST_CASE("default phase speed includes kinetic moment contributions")
{
  const auto                        params     = parameters();
  const hybrid::engine::FluidState  fluid      = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                  0.8, -0.1, 0.5,  -0.3, 0.6};
  const hybrid::engine::FieldState  field      = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const hybrid::engine::VectorState background = {0.2, -0.1, 0.3};
  const std::vector<hybrid::engine::KineticPhaseMoment> kinetic = {
      {{{0.15, 0.02, -0.03, 0.04, 0.12, 0.10, 0.08, 0, 0, 0}}, -0.5},
      {{{0.25, -0.04, 0.05, -0.02, 0.2, 0.16, 0.18, 0, 0, 0}}, 0.75}};

  const auto phase = hybrid::engine::default_phase_speed(fluid, field, background, kinetic, params);
  const hybrid::engine::PhaseState expected = {
      1.4256419559206712, 1.235904045252528,  1.126933945584754,
      1.583886411603795,  1.3287369099426654, 1.1287369099426654,
      1.5326549065840962, 1.4007981328668573, 1.1336884983153623};
  for (int component = 0; component < hybrid::num_phase_directions * hybrid::num_phase_branches;
       ++component) {
    REQUIRE(phase[component] == Catch::Approx(expected[component]).epsilon(tolerance));
  }
}

TEST_CASE("phase cell-to-face interpolation uses directional maxima")
{
  const hybrid::engine::PhaseState left  = {1.0, 2.0, 3.0, 4.0, 1.5, 6.0, 7.0, 8.0, 9.0};
  const hybrid::engine::PhaseState right = {1.5, 1.0, 0.5, 3.0, 2.5, 0.25, 5.0, 9.0, 0.75};

  const auto x_face = hybrid::engine::phase_cell_to_face(left, right, 0);
  REQUIRE(x_face[0] == Catch::Approx(1.5));
  REQUIRE(x_face[1] == Catch::Approx(2.0));
  REQUIRE(x_face[2] == 0.0);

  const auto y_face = hybrid::engine::phase_cell_to_face(left, right, 1);
  REQUIRE(y_face[3] == Catch::Approx(4.0));
  REQUIRE(y_face[4] == Catch::Approx(2.5));
  REQUIRE(y_face[5] == 0.0);

  const auto z_face = hybrid::engine::phase_cell_to_face(left, right, 2);
  REQUIRE(z_face[6] == Catch::Approx(7.0));
  REQUIRE(z_face[7] == Catch::Approx(9.0));
  REQUIRE(z_face[8] == 0.0);
  REQUIRE_THROWS_AS(hybrid::engine::phase_cell_to_face(left, right, 3), std::out_of_range);
}
