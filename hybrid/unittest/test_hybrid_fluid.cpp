// -*- C++ -*-
#include "engine/fluid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;

hybrid::engine::FluidParameters parameters()
{
  return {20.0, 5.0 / 3.0, -2.0, 1.0, 0.7};
}
} // namespace

TEST_CASE("fluid primitive and conservative states round trip")
{
  auto                       params = parameters();
  hybrid::engine::FluidState fluid  = {1.2, 0.3, -0.2, 0.4, 0, 0.8, -0.1, 0.5, -0.3, 0.6};
  fluid[4] = params.electron_entropy * std::pow(fluid[0], params.adiabatic_index);
  const hybrid::engine::FieldState  field     = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const auto                        conserved = hybrid::engine::conservative(fluid, field, params);
  const hybrid::engine::VectorState curl_b    = {0.08, -0.04, 0.03};
  const hybrid::engine::ConservedState expected_conserved = {
      2.0, 0.27999999999999997, 0.16000000000000003, 0.24, 2.7108535242058465};
  for (int component = 0; component < hybrid::num_conserved_components; ++component) {
    REQUIRE(conserved[component] ==
            Catch::Approx(expected_conserved[component]).epsilon(tolerance));
  }

  const hybrid::engine::CurrentState current = {1.5999999999999996, 0.8799999999999999,
                                                -0.9199999999999999, 1.23};
  const auto                         recovered =
      hybrid::engine::primitive(expected_conserved, field, curl_b, current, params);
  for (int component = 0; component < hybrid::num_fluid_components; ++component) {
    REQUIRE(recovered[component] == Catch::Approx(fluid[component]).epsilon(tolerance));
  }
}

TEST_CASE("standard hybrid primitive recovery leaves the ion fluid inactive")
{
  auto params                                    = parameters();
  params.ion_charge_to_mass                      = 0;
  const hybrid::engine::ConservedState conserved = {3, 1.7, -2.1, 0.9, 5};
  const hybrid::engine::FieldState     field     = {};
  const hybrid::engine::VectorState    curl_b    = {};
  const hybrid::engine::CurrentState   current   = {4, 0.8, -0.4, 0.2};
  const auto fluid = hybrid::engine::primitive(conserved, field, curl_b, current, params);
  for (int component = 5; component < 9; ++component) {
    REQUIRE(fluid[component] == 0);
  }
  REQUIRE(fluid[9] == Catch::Approx(1.0759718605778539).epsilon(tolerance));
}

TEST_CASE("fluid RHS includes background magnetic force but not background electric work")
{
  const auto                         params     = parameters();
  const hybrid::engine::FieldState   field      = {1, 2, -1, 0.5, -0.25, 0.75};
  const hybrid::engine::CurrentState current    = {0.4, -0.2, 0.3, -0.5};
  const hybrid::engine::VectorState  background = {1.0, 0.5, -0.5};
  const auto rhs = hybrid::engine::fluid_rhs(0.2, field, current, background, params);
  REQUIRE(rhs[0] == 0);
  REQUIRE(rhs[1] == Catch::Approx(-0.08200000000000002));
  REQUIRE(rhs[2] == Catch::Approx(-0.15300000000000002));
  REQUIRE(rhs[3] == Catch::Approx(0.08500000000000002));
  REQUIRE(rhs[4] == Catch::Approx(-0.18));
}

TEST_CASE("directional fluid flux rotates consistently")
{
  const auto                        params     = parameters();
  const hybrid::engine::FluidState  fluid      = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                  0.8, -0.1, 0.5,  -0.3, 0.6};
  const hybrid::engine::FieldState  field      = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const hybrid::engine::VectorState background = {0.2, -0.1, 0.3};
  const std::array<hybrid::engine::ConservedState, 3> expected = {
      hybrid::engine::ConservedState{0.27999999999999997, 1.6398953009328328, -0.03083097902313338,
                                     0.10035914918594448, 0.52004576628811},
      hybrid::engine::ConservedState{0.16000000000000003, -0.03083097902313338, 1.8530643219096994,
                                     -0.17382394008064772, 0.35866432437153023},
      hybrid::engine::ConservedState{0.24, 0.10035914918594448, -0.17382394008064772,
                                     1.8873671403652674, 0.8285587089866779}};
  for (int direction = 0; direction < 3; ++direction) {
    const auto flux = hybrid::engine::physical_flux(direction, fluid, field, background, params);
    for (int component = 0; component < hybrid::num_conserved_components; ++component) {
      REQUIRE(flux[component] == Catch::Approx(expected[direction][component]).epsilon(tolerance));
    }
  }

  const hybrid::engine::VectorState zero = {};
  const auto no_background = hybrid::engine::physical_flux(0, fluid, field, zero, params);
  REQUIRE(no_background[0] ==
          Catch::Approx(fluid[5] * fluid[6] + fluid[0] * fluid[1]).margin(1.0e-14));
  REQUIRE_THROWS_AS(hybrid::engine::physical_flux(3, fluid, field, zero, params),
                    std::out_of_range);
}
