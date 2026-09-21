// -*- C++ -*-
#ifndef _HYBRID_ENGINE_PCC2_HPP_
#define _HYBRID_ENGINE_PCC2_HPP_

#include "hybrid.hpp"

namespace hybrid::engine
{
enum class Pcc2Stage {
  Idle,
  PredictorField,
  PredictorOhm,
  PredictorAverage,
  FirstCorrectorParticle,
  FirstCorrectorMoment,
  FirstCorrectorField,
  FirstCorrectorOhm,
  FirstCorrectorAverage,
  SecondCorrectorParticle,
  SecondCorrectorMoment,
  SecondCorrectorField,
  SecondCorrectorOhm,
  Commit,
};

inline constexpr int pcc2_stage_count()
{
  return static_cast<int>(Pcc2Stage::Commit) + 1;
}

struct Pcc2State {
  Pcc2Stage stage = Pcc2Stage::Idle;
  int       substep;
};

inline const char* pcc2_stage_name(Pcc2Stage stage)
{
  switch (stage) {
  case Pcc2Stage::Idle:
    return "Idle";
  case Pcc2Stage::PredictorField:
    return "PredictorField";
  case Pcc2Stage::PredictorOhm:
    return "PredictorOhm";
  case Pcc2Stage::PredictorAverage:
    return "PredictorAverage";
  case Pcc2Stage::FirstCorrectorParticle:
    return "FirstCorrectorParticle";
  case Pcc2Stage::FirstCorrectorMoment:
    return "FirstCorrectorMoment";
  case Pcc2Stage::FirstCorrectorField:
    return "FirstCorrectorField";
  case Pcc2Stage::FirstCorrectorOhm:
    return "FirstCorrectorOhm";
  case Pcc2Stage::FirstCorrectorAverage:
    return "FirstCorrectorAverage";
  case Pcc2Stage::SecondCorrectorParticle:
    return "SecondCorrectorParticle";
  case Pcc2Stage::SecondCorrectorMoment:
    return "SecondCorrectorMoment";
  case Pcc2Stage::SecondCorrectorField:
    return "SecondCorrectorField";
  case Pcc2Stage::SecondCorrectorOhm:
    return "SecondCorrectorOhm";
  case Pcc2Stage::Commit:
    return "Commit";
  }
  return "Unknown";
}

inline Pcc2Stage pcc2_next_stage(Pcc2Stage stage)
{
  const int next = static_cast<int>(stage) + 1;
  if (next >= pcc2_stage_count()) {
    return Pcc2Stage::Idle;
  }
  return static_cast<Pcc2Stage>(next);
}

inline bool pcc2_is_field_stage(Pcc2Stage stage)
{
  return stage == Pcc2Stage::PredictorField || stage == Pcc2Stage::FirstCorrectorField ||
         stage == Pcc2Stage::SecondCorrectorField;
}

inline bool pcc2_is_ohm_stage(Pcc2Stage stage)
{
  return stage == Pcc2Stage::PredictorOhm || stage == Pcc2Stage::FirstCorrectorOhm ||
         stage == Pcc2Stage::SecondCorrectorOhm;
}

inline bool pcc2_is_particle_stage(Pcc2Stage stage)
{
  return stage == Pcc2Stage::FirstCorrectorParticle || stage == Pcc2Stage::SecondCorrectorParticle;
}

inline bool pcc2_is_average_stage(Pcc2Stage stage)
{
  return stage == Pcc2Stage::PredictorAverage || stage == Pcc2Stage::FirstCorrectorAverage;
}

inline bool pcc2_should_rollback_particles(Pcc2Stage stage)
{
  return stage == Pcc2Stage::FirstCorrectorParticle;
}
} // namespace hybrid::engine

#endif
