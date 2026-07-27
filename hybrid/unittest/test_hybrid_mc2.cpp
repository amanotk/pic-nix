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
}

TEST_CASE("HLL edge electric formulas preserve legacy signs")
{
  REQUIRE(hybrid::engine::hll_edge_electric_positive(0.6, -0.2, -0.3, 1.4, 1.7, 1.2, 20.0) ==
          Catch::Approx(0.19427586206896552).epsilon(tolerance));
  REQUIRE(hybrid::engine::hll_edge_electric_negative(0.7, -0.4, -0.8, 0.9, 1.7, 1.2, 20.0) ==
          Catch::Approx(0.06262068965517241).epsilon(tolerance));
}
