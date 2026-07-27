// -*- C++ -*-
#include "engine/pcc2.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PCC2 stage enumeration covers the legacy predictor-corrector flow")
{
  REQUIRE(hybrid::engine::pcc2_stage_count() == 14);
  REQUIRE(hybrid::engine::pcc2_is_field_stage(hybrid::engine::Pcc2Stage::PredictorField));
  REQUIRE(hybrid::engine::pcc2_is_field_stage(hybrid::engine::Pcc2Stage::FirstCorrectorField));
  REQUIRE(hybrid::engine::pcc2_is_field_stage(hybrid::engine::Pcc2Stage::SecondCorrectorField));
  REQUIRE_FALSE(hybrid::engine::pcc2_is_field_stage(hybrid::engine::Pcc2Stage::PredictorOhm));

  REQUIRE(hybrid::engine::pcc2_is_ohm_stage(hybrid::engine::Pcc2Stage::PredictorOhm));
  REQUIRE(hybrid::engine::pcc2_is_ohm_stage(hybrid::engine::Pcc2Stage::FirstCorrectorOhm));
  REQUIRE(hybrid::engine::pcc2_is_ohm_stage(hybrid::engine::Pcc2Stage::SecondCorrectorOhm));
  REQUIRE_FALSE(hybrid::engine::pcc2_is_ohm_stage(hybrid::engine::Pcc2Stage::PredictorField));

  REQUIRE(
      hybrid::engine::pcc2_is_particle_stage(hybrid::engine::Pcc2Stage::FirstCorrectorParticle));
  REQUIRE(
      hybrid::engine::pcc2_is_particle_stage(hybrid::engine::Pcc2Stage::SecondCorrectorParticle));
  REQUIRE_FALSE(hybrid::engine::pcc2_is_particle_stage(hybrid::engine::Pcc2Stage::PredictorField));

  REQUIRE(hybrid::engine::pcc2_is_average_stage(hybrid::engine::Pcc2Stage::PredictorAverage));
  REQUIRE(hybrid::engine::pcc2_is_average_stage(hybrid::engine::Pcc2Stage::FirstCorrectorAverage));
  REQUIRE_FALSE(hybrid::engine::pcc2_is_average_stage(hybrid::engine::Pcc2Stage::Commit));

  REQUIRE(hybrid::engine::pcc2_should_rollback_particles(
      hybrid::engine::Pcc2Stage::FirstCorrectorParticle));
  REQUIRE_FALSE(hybrid::engine::pcc2_should_rollback_particles(
      hybrid::engine::Pcc2Stage::SecondCorrectorParticle));
}

TEST_CASE("PCC2 stage transition follows the correct order")
{
  using P = hybrid::engine::Pcc2Stage;
  REQUIRE(hybrid::engine::pcc2_next_stage(P::Idle) == P::PredictorField);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::PredictorField) == P::PredictorOhm);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::PredictorOhm) == P::PredictorAverage);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::PredictorAverage) == P::FirstCorrectorParticle);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::FirstCorrectorParticle) == P::FirstCorrectorMoment);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::FirstCorrectorMoment) == P::FirstCorrectorField);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::FirstCorrectorField) == P::FirstCorrectorOhm);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::FirstCorrectorOhm) == P::FirstCorrectorAverage);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::FirstCorrectorAverage) == P::SecondCorrectorParticle);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::SecondCorrectorParticle) == P::SecondCorrectorMoment);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::SecondCorrectorMoment) == P::SecondCorrectorField);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::SecondCorrectorField) == P::SecondCorrectorOhm);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::SecondCorrectorOhm) == P::Commit);
  REQUIRE(hybrid::engine::pcc2_next_stage(P::Commit) == P::Idle);
}

TEST_CASE("PCC2 state tracks substep counter")
{
  hybrid::engine::Pcc2State state = {};
  REQUIRE(state.stage == hybrid::engine::Pcc2Stage::Idle);

  state.stage   = hybrid::engine::Pcc2Stage::PredictorField;
  state.substep = 1;
  REQUIRE(std::string(hybrid::engine::pcc2_stage_name(state.stage)) == "PredictorField");

  state.stage = hybrid::engine::pcc2_next_stage(state.stage);
  REQUIRE(state.stage == hybrid::engine::Pcc2Stage::PredictorOhm);

  state.stage = hybrid::engine::Pcc2Stage::Commit;
  REQUIRE(state.stage == hybrid::engine::Pcc2Stage::Commit);
}
