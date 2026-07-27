// -*- C++ -*-
#include "engine/ohm_solver.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;
}

TEST_CASE("Ohm source constructs cross-B and pressure divergence terms")
{
  const std::array<nix::float64, hybrid::num_moment_components> moment = {
      1.2, 0.3, -0.2, 0.4, 0.12, 0.10, 0.08, 0.01, 0.02, -0.01};
  const std::array<nix::float64, hybrid::num_field_components>  field       = {0.2, -0.3, 0.1,
                                                                               0.8, 1.2,  -0.4};
  const hybrid::engine::VectorState                             background  = {0.2, -0.1, 0.3};
  const std::array<nix::float64, hybrid::num_moment_components> plus_x_plus = {
      1.3, 0.4, -0.1, 0.5, 0.15, 0.11, 0.09, 0.02, 0.03, 0.0};
  const std::array<nix::float64, hybrid::num_moment_components> plus_x_minus = {
      1.1, 0.2, -0.3, 0.3, 0.09, 0.09, 0.07, 0.0, 0.01, -0.02};
  const std::array<nix::float64, hybrid::num_moment_components> plus_y_plus = {
      1.25, 0.35, -0.15, 0.45, 0.14, 0.12, 0.10, 0.03, 0.04, 0.01};
  const std::array<nix::float64, hybrid::num_moment_components> plus_y_minus = {
      1.15, 0.25, -0.25, 0.35, 0.10, 0.08, 0.06, -0.01, 0.0, -0.03};
  const std::array<nix::float64, hybrid::num_moment_components> plus_z_plus = {
      1.22, 0.32, -0.22, 0.42, 0.13, 0.11, 0.09, 0.015, 0.025, -0.005};
  const std::array<nix::float64, hybrid::num_moment_components> plus_z_minus = {
      1.18, 0.28, -0.18, 0.38, 0.11, 0.09, 0.07, 0.005, 0.015, -0.015};

  const auto src = hybrid::engine::construct_ohm_source(moment, field, background, 20.0, 0.5, 2.0,
                                                        4.0, plus_x_plus, plus_x_minus, plus_y_plus,
                                                        plus_y_minus, plus_z_plus, plus_z_minus);
  const hybrid::engine::OhmSource expected = {1.2, 0.09225, 0.009749999999999995,
                                              0.005999999999999998};
  for (int component = 0; component < hybrid::num_ohm_source_components; ++component) {
    REQUIRE(src[component] == Catch::Approx(expected[component]).epsilon(1.0e-12));
  }
}

TEST_CASE("resistive field includes eta and hyper-resistive chi terms")
{
  const hybrid::engine::VectorState rot    = {0.05, -0.02, 0.03};
  const hybrid::engine::VectorState rot_xm = {0.04, -0.02, 0.03};
  const hybrid::engine::VectorState rot_xp = {0.06, -0.02, 0.03};
  const hybrid::engine::VectorState rot_ym = {0.05, -0.03, 0.03};
  const hybrid::engine::VectorState rot_yp = {0.05, -0.01, 0.03};
  const hybrid::engine::VectorState rot_zm = {0.05, -0.02, 0.02};
  const hybrid::engine::VectorState rot_zp = {0.05, -0.02, 0.04};

  const auto res = hybrid::engine::resistive_field(rot, rot_xm, rot_xp, rot_ym, rot_yp, rot_zm,
                                                   rot_zp, 0.01, 0.001, 0.5, 2.0, 4.0);
  const hybrid::engine::VectorState expected = {0.0005, -0.0002, 0.0003};
  for (int direction = 0; direction < hybrid::num_vector_components; ++direction) {
    REQUIRE(res[direction] == Catch::Approx(expected[direction]).epsilon(tolerance));
  }
}

TEST_CASE("SSOR2 coefficients match legacy light-speed scaled Laplacian")
{
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  REQUIRE(coeff.laplacian_x == Catch::Approx(1600.0));
  REQUIRE(coeff.laplacian_y == Catch::Approx(100.0));
  REQUIRE(coeff.laplacian_z == Catch::Approx(25.0));
  REQUIRE(coeff.diagonal_minus_2_laplacian == Catch::Approx(3450.0));
}

TEST_CASE("SSOR2 update produces legacy relaxed solution")
{
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  const hybrid::engine::OhmSource src = {1.2, 0.09225, 0.009749999999999995, 0.005999999999999998};
  const std::array<nix::float64, 3> left  = {0.1, -0.2, 0.05};
  const std::array<nix::float64, 3> right = {0.3, -0.1, 0.15};
  const std::array<nix::float64, 3> back  = {0.15, -0.15, 0.08};
  const std::array<nix::float64, 3> front = {0.25, -0.12, 0.12};
  const std::array<nix::float64, 3> down  = {0.18, -0.18, 0.07};
  const std::array<nix::float64, 3> up    = {0.22, -0.16, 0.13};

  const std::array<nix::float64, 3> expected = {0.19995718880389432, -0.14936551054705613,
                                                0.09996696801112656};
  for (int component = 0; component < 3; ++component) {
    const auto value = hybrid::engine::ssor2_update(
        src, left[component], right[component], back[component], front[component], down[component],
        up[component], coeff, component + hybrid::current_component::current_x);
    REQUIRE(value == Catch::Approx(expected[component]).epsilon(tolerance));
  }
}

TEST_CASE("SSOR2 residual matches legacy error formula")
{
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  const hybrid::engine::OhmSource src  = {1.2, 0.09225, 0.009749999999999995, 0.005999999999999998};
  const std::array<nix::float64, 3> eb = {0.2, -0.15, 0.1};
  const std::array<nix::float64, 3> left  = {0.1, -0.2, 0.05};
  const std::array<nix::float64, 3> right = {0.3, -0.1, 0.15};
  const std::array<nix::float64, 3> back  = {0.15, -0.15, 0.08};
  const std::array<nix::float64, 3> front = {0.25, -0.12, 0.12};
  const std::array<nix::float64, 3> down  = {0.18, -0.18, 0.07};
  const std::array<nix::float64, 3> up    = {0.22, -0.16, 0.13};

  const std::array<nix::float64, 3> expected = {0.14774999999997362, -2.18974999999989,
                                                0.11400000000003274};
  for (int component = 0; component < 3; ++component) {
    const auto residual = hybrid::engine::ssor2_residual(
        src, eb[component], left[component], right[component], back[component], front[component],
        down[component], up[component], coeff, component + hybrid::current_component::current_x);
    REQUIRE(residual == Catch::Approx(expected[component]).epsilon(1.0e-12));
  }
}

TEST_CASE("SSOR2 convergence check uses relative error tolerance")
{
  REQUIRE_FALSE(hybrid::engine::ssor2_converged(100.0, 1.0, 1.0e-5, 5, 100));
  REQUIRE(hybrid::engine::ssor2_converged(1.0e-12, 1.0, 1.0e-5, 5, 100));
  REQUIRE(hybrid::engine::ssor2_converged(100.0, 1.0, 1.0e-5, 100, 100));
}
