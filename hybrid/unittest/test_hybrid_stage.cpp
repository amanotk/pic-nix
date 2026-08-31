// -*- C++ -*-
#include "engine/field.hpp"
#include "engine/fluid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;
}

TEST_CASE("fluid conservative update matches legacy PCC2 flux divergence")
{
  const hybrid::engine::ConservedState baseline     = {2.0, 0.28, 0.16, 0.24, 2.7108535242058465};
  const hybrid::engine::ConservedState rhs          = {0, -0.082, -0.153, 0.085, -0.18};
  const hybrid::engine::ConservedState flux_x_minus = {0.01, -0.02, 0.03, -0.04, 0.05};
  const hybrid::engine::ConservedState flux_x_plus  = {-0.03, 0.04, -0.05, 0.06, -0.07};
  const hybrid::engine::ConservedState flux_y_minus = {0.02, 0.01, -0.04, 0.03, -0.02};
  const hybrid::engine::ConservedState flux_y_plus  = {0.05, -0.02, 0.02, -0.01, 0.04};
  const hybrid::engine::ConservedState flux_z_minus = {-0.01, 0.03, 0.02, -0.05, 0.06};
  const hybrid::engine::ConservedState flux_z_plus  = {0.04, -0.01, -0.03, 0.02, -0.08};

  const auto updated = hybrid::engine::advance_conserved_fluid(
      baseline, flux_x_minus, flux_x_plus, flux_y_minus, flux_y_plus, flux_z_minus, flux_z_plus,
      rhs, {0.5, 2.0, 4.0});
  const hybrid::engine::ConservedState expected = {2.0524999999999998, 0.10300000000000005,
                                                   0.14950000000000005, 0.1275, 2.775853524205847};
  for (int component = 0; component < hybrid::num_conserved_components; ++component) {
    REQUIRE(updated[component] == Catch::Approx(expected[component]).epsilon(tolerance));
  }
}

TEST_CASE("local field and fluid stage formulas compose without hidden time factors")
{
  const hybrid::engine::GridSpacing spacing = {0.5, 2.0, 4.0};
  hybrid::engine::FieldState        baseline{};
  baseline[hybrid::field_component::magnetic_x] = 1.2;
  baseline[hybrid::field_component::magnetic_y] = -0.7;
  baseline[hybrid::field_component::magnetic_z] = 0.4;
  const hybrid::engine::FieldState edge         = {0.9, -1.1, 0.6, 0, 0, 0};
  const hybrid::engine::FieldState x_minus      = {0.0, -0.2, -0.8, 0, 0, 0};
  const hybrid::engine::FieldState y_minus      = {-0.3, 0.4, 0.1, 0, 0, 0};
  const hybrid::engine::FieldState z_minus      = {0.2, -0.6, 0.5, 0, 0, 0};

  const auto magnetic = hybrid::engine::constrained_transport_magnetic(
      baseline, edge, x_minus, y_minus, z_minus, spacing, 20.0, 0.05);
  REQUIRE(magnetic[0] == Catch::Approx(0.825).epsilon(tolerance));
  REQUIRE(magnetic[1] == Catch::Approx(1.925).epsilon(tolerance));
  REQUIRE(magnetic[2] == Catch::Approx(2.8).epsilon(tolerance));
}

TEST_CASE("local field and fluid stage recovers primitive variables")
{
  const hybrid::engine::ConservedState  updated    = {2.0524999999999998, 0.10300000000000005,
                                                      0.14950000000000005, 0.1275, 2.775853524205847};
  const hybrid::engine::FieldState      field      = {0.2, -0.3, 0.1, 0.8, 1.2, -0.4};
  const hybrid::engine::VectorState     curl_b     = {0.05, -0.02, 0.03};
  const hybrid::engine::CurrentState    current    = {1.4, 0.2, -0.5, 0.7};
  const hybrid::engine::FluidParameters parameters = {20.0, 5.0 / 3.0, -2.0, 1.0, 0.7};

  const auto primitive = hybrid::engine::primitive(updated, field, curl_b, current, parameters);
  const hybrid::engine::FluidState expected = {
      1.1508333333333332,  0.07328023171614774, -0.0957277335264301,  0.2309920347574221,
      0.8846784076348351,  0.9016666666666666,  0.020702402957486182, 0.2879852125693161,
      -0.1534195933456561, 0.8482990954515024};
  for (int component = 0; component < hybrid::num_fluid_components; ++component) {
    REQUIRE(primitive[component] == Catch::Approx(expected[component]).epsilon(tolerance));
  }
}
