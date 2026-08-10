// -*- C++ -*-
#ifndef _PIC_PERFORMANCE_HPP_
#define _PIC_PERFORMANCE_HPP_

#include "nix/nix.hpp"

#include <array>
#include <vector>

class PicPerformance
{
public:
  enum class Phase {
    Advance,
    CurrentField,
    ParticleProbe,
    ParticleExchange,
    FieldExchange,
    Count,
  };

  enum class Operation {
    CurrentBegin,
    ParticleBegin,
    CurrentWaitall,
    FieldBegin,
    ParticleProbe,
    ParticleWaitall,
    FieldWaitall,
    CurrentPoll,
    ParticlePoll,
    FieldPoll,
    Count,
  };

  static constexpr int NumPhases     = static_cast<int>(Phase::Count);
  static constexpr int NumOperations = static_cast<int>(Operation::Count);

  bool configure(const nix::json& application);

  bool begin_step(int step, int nthread);

  void set_parallel_threads(int nthread);

  bool is_sampling() const
  {
    return sampling;
  }

  void record_chunk(Phase phase, nix::float64 elapsed);

  void record_phase_wall(Phase phase, nix::float64 elapsed);

  void record_operation(Operation operation, nix::float64 elapsed);

  void record_operation_summary(Operation operation, nix::float64 total, nix::float64 max_call);

  // begin/end pairs: measure a region with a single call at each end. The
  // wall-clock read and the recording are skipped entirely when the sampler
  // is inactive, so call sites need no sampling branches.
  void begin_chunk(Phase phase);

  void end_chunk(Phase phase);

  void begin_wall(Phase phase);

  void end_wall(Phase phase);

  void begin_operation(Operation operation);

  void end_operation(Operation operation);

  nix::json finish_step(nix::float64 local_push, nix::float64 barrier_wait,
                        MPI_Comm comm = MPI_COMM_WORLD);

private:
  struct ThreadTiming {
    std::array<nix::float64, NumPhases>     begin_chunk{};
    std::array<nix::float64, NumPhases>     begin_wall{};
    std::array<nix::float64, NumOperations> begin_operation{};
    std::array<nix::float64, NumPhases>     busy{};
    std::array<nix::float64, NumPhases>     max_chunk{};
    std::array<nix::float64, NumPhases>     wall{};
    std::array<nix::float64, NumOperations> operation_total{};
    std::array<nix::float64, NumOperations> operation_max_call{};
  };

  bool                      enabled          = false;
  bool                      sampling         = false;
  int                       interval         = 100;
  int                       offset           = 0;
  int                       parallel_threads = 1;
  std::vector<ThreadTiming> thread_timing;

  static int phase_index(Phase phase)
  {
    return static_cast<int>(phase);
  }

  static int operation_index(Operation operation)
  {
    return static_cast<int>(operation);
  }

  static nix::json summarize(const std::vector<nix::float64>& values);
};

#endif
