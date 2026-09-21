// -*- C++ -*-
#include "engine/field.hpp"
#include "engine/fluid.hpp"
#include "engine/pcc2.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;

hybrid::engine::FluidParameters fluid_params()
{
  return {20.0, 5.0 / 3.0, -2.0, 1.0, 0.7};
}
} // namespace

TEST_CASE("PCC2 stage flow preserves uniform conservative update trajectory")
{
  const hybrid::engine::FluidState   accepted_fluid = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                       0.8, -0.1, 0.5,  -0.3, 0.6};
  const hybrid::engine::FieldState   accepted_field = {0.2, -0.3, 0.1, 1.1, -0.7, 0.4};
  const hybrid::engine::VectorState  background     = {0.2, -0.1, 0.3};
  const hybrid::engine::CurrentState current        = {0.4, -0.2, 0.3, -0.5};
  const auto                         params         = fluid_params();
  const nix::float64                 dt             = 0.05;

  const auto accepted_cons = hybrid::engine::conservative(accepted_fluid, accepted_field, params);
  const auto rhs = hybrid::engine::fluid_rhs(dt, accepted_field, current, background, params);

  const hybrid::engine::ConservedState expected_accepted = {
      2.0, 0.27999999999999997, 0.16000000000000003, 0.24, 2.7108535242058465};
  for (int c = 0; c < hybrid::num_conserved_components; ++c) {
    REQUIRE(accepted_cons[c] == Catch::Approx(expected_accepted[c]).epsilon(tolerance));
  }

  // All three field stages produce identical working = accepted + rhs
  hybrid::engine::ConservedState working = {};
  for (int c = 0; c < hybrid::num_conserved_components; ++c) {
    working[c] = accepted_cons[c] + rhs[c];
  }
  const hybrid::engine::ConservedState expected_working = {
      2.0, 0.27647499999999997, 0.16727500000000003, 0.23857499999999998, 2.7198535242058464};
  for (int c = 0; c < hybrid::num_conserved_components; ++c) {
    REQUIRE(working[c] == Catch::Approx(expected_working[c]).epsilon(tolerance));
  }

  // Average = 50/50 between working and accepted
  hybrid::engine::ConservedState averaged = {};
  for (int c = 0; c < hybrid::num_conserved_components; ++c) {
    averaged[c] = 0.5 * (working[c] + accepted_cons[c]);
  }
  const hybrid::engine::ConservedState expected_averaged = {
      2.0, 0.27823749999999997, 0.16363750000000005, 0.2392875, 2.715353524205846};
  for (int c = 0; c < hybrid::num_conserved_components; ++c) {
    REQUIRE(averaged[c] == Catch::Approx(expected_averaged[c]).epsilon(tolerance));
  }

  // Stage count matches legacy
  using P             = hybrid::engine::Pcc2Stage;
  int field_stages    = 0;
  int ohm_stages      = 0;
  int average_stages  = 0;
  int particle_stages = 0;
  for (P stage = P::PredictorField; stage != P::Idle;
       stage   = hybrid::engine::pcc2_next_stage(stage)) {
    if (hybrid::engine::pcc2_is_field_stage(stage))
      ++field_stages;
    if (hybrid::engine::pcc2_is_ohm_stage(stage))
      ++ohm_stages;
    if (hybrid::engine::pcc2_is_average_stage(stage))
      ++average_stages;
    if (hybrid::engine::pcc2_is_particle_stage(stage))
      ++particle_stages;
    if (stage == P::Commit)
      break;
  }
  REQUIRE(field_stages == 3);
  REQUIRE(ohm_stages == 3);
  REQUIRE(average_stages == 2);
  REQUIRE(particle_stages == 2);
  REQUIRE(hybrid::engine::pcc2_should_rollback_particles(P::FirstCorrectorParticle));
}

TEST_CASE("PCC2 fluid update uses accepted conservation and working-field forcing")
{
  const hybrid::engine::FluidState   fluid = {1.0, 0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.3, 0.2, 0.6};
  const hybrid::engine::FieldState   accepted_field = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0};
  const hybrid::engine::FieldState   working_field  = {2.0, -1.0, 0.5, 0.0, 3.0, -2.0};
  const hybrid::engine::CurrentState current        = {0.4, -0.3, 0.2, -0.1};
  const hybrid::engine::VectorState  background     = {0.2, 0.1, -0.1};
  const auto                         params         = fluid_params();
  const auto baseline = hybrid::engine::conservative(fluid, accepted_field, params);
  const auto forcing  = hybrid::engine::fluid_rhs(0.05, working_field, current, background, params);

  REQUIRE(baseline[4] == Catch::Approx(0.1425 + 2.1 + 0.5 / nix::math::pi4));
  REQUIRE(forcing[1] == Catch::Approx(-0.039725));
  REQUIRE(forcing[2] == Catch::Approx(0.021625));
  REQUIRE(forcing[3] == Catch::Approx(-0.007575));
  REQUIRE(forcing[4] == Catch::Approx(0.0425));
  const auto accepted_forcing =
      hybrid::engine::fluid_rhs(0.05, accepted_field, current, background, params);
  REQUIRE(forcing != accepted_forcing);
}
