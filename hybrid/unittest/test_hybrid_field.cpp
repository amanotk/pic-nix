// -*- C++ -*-
#include "engine/field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;
}

TEST_CASE("magnetic interpolation preserves legacy face staggering")
{
  REQUIRE(hybrid::engine::magnetic_cell_to_face(hybrid::field_component::magnetic_x, 1.25, -0.75) ==
          Catch::Approx(0.25));
  REQUIRE(hybrid::engine::magnetic_face_to_cell(hybrid::field_component::magnetic_z, -2.0, 0.5) ==
          Catch::Approx(-0.75));
  REQUIRE_THROWS_AS(
      hybrid::engine::magnetic_cell_to_face(hybrid::field_component::electric_x, 0, 0),
      std::out_of_range);
}

TEST_CASE("edge electric average matches legacy four-cell stencil")
{
  REQUIRE(hybrid::engine::edge_electric_average(1.0, -0.5, 2.5, 4.0) == Catch::Approx(1.75));
}

TEST_CASE("cell-centered curl uses legacy c over four pi scaling")
{
  const hybrid::engine::GridSpacing spacing = {0.5, 2.0, 4.0};
  hybrid::engine::FieldState        x_plus  = {};
  hybrid::engine::FieldState        x_minus = {};
  hybrid::engine::FieldState        y_plus  = {};
  hybrid::engine::FieldState        y_minus = {};
  hybrid::engine::FieldState        z_plus  = {};
  hybrid::engine::FieldState        z_minus = {};

  x_plus[hybrid::field_component::magnetic_y]  = 1.5;
  x_minus[hybrid::field_component::magnetic_y] = -0.5;
  x_plus[hybrid::field_component::magnetic_z]  = 1.3;
  x_minus[hybrid::field_component::magnetic_z] = 0.1;
  y_plus[hybrid::field_component::magnetic_x]  = 0.7;
  y_minus[hybrid::field_component::magnetic_x] = -0.1;
  y_plus[hybrid::field_component::magnetic_z]  = 1.7;
  y_minus[hybrid::field_component::magnetic_z] = -0.5;
  z_plus[hybrid::field_component::magnetic_x]  = 0.8;
  z_minus[hybrid::field_component::magnetic_x] = -0.4;
  z_plus[hybrid::field_component::magnetic_y]  = 0.9;
  z_minus[hybrid::field_component::magnetic_y] = -1.1;

  const auto curl = hybrid::engine::curl_magnetic(x_plus, x_minus, y_plus, y_minus, z_plus, z_minus,
                                                  spacing, 20.0);
  REQUIRE(curl[0] == Catch::Approx(6.0 / nix::math::pi4).epsilon(tolerance));
  REQUIRE(curl[1] == Catch::Approx(-21.0 / nix::math::pi4).epsilon(tolerance));
  REQUIRE(curl[2] == Catch::Approx(36.0 / nix::math::pi4).epsilon(tolerance));
}

TEST_CASE("constrained transport updates face magnetic field from edge electric curl")
{
  const hybrid::engine::GridSpacing spacing = {0.5, 2.0, 4.0};
  hybrid::engine::FieldState        baseline{};
  baseline[hybrid::field_component::magnetic_x] = 1.2;
  baseline[hybrid::field_component::magnetic_y] = -0.7;
  baseline[hybrid::field_component::magnetic_z] = 0.4;

  const hybrid::engine::FieldState edge    = {0.9, -1.1, 0.6, 0, 0, 0};
  const hybrid::engine::FieldState x_minus = {0.0, -0.2, -0.8, 0, 0, 0};
  const hybrid::engine::FieldState y_minus = {-0.3, 0.4, 0.1, 0, 0, 0};
  const hybrid::engine::FieldState z_minus = {0.2, -0.6, 0.5, 0, 0, 0};

  const auto magnetic = hybrid::engine::constrained_transport_magnetic(
      baseline, edge, x_minus, y_minus, z_minus, spacing, 20.0, 0.05);
  REQUIRE(magnetic[0] == Catch::Approx(0.825).epsilon(tolerance));
  REQUIRE(magnetic[1] == Catch::Approx(1.925).epsilon(tolerance));
  REQUIRE(magnetic[2] == Catch::Approx(2.8).epsilon(tolerance));

  const auto uniform = hybrid::engine::constrained_transport_magnetic(baseline, edge, edge, edge,
                                                                      edge, spacing, 20.0, 0.05);
  REQUIRE(uniform[0] == Catch::Approx(baseline[hybrid::field_component::magnetic_x]));
  REQUIRE(uniform[1] == Catch::Approx(baseline[hybrid::field_component::magnetic_y]));
  REQUIRE(uniform[2] == Catch::Approx(baseline[hybrid::field_component::magnetic_z]));
}
