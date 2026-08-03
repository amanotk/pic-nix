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

  static constexpr int NumPhases = static_cast<int>(Phase::Count);

  bool configure(const nix::json& application);

  bool begin_step(int step, int nthread);

  void set_parallel_threads(int nthread);

  bool is_sampling() const
  {
    return sampling;
  }

  void record_chunk(Phase phase, nix::float64 elapsed);

  void record_phase_wall(Phase phase, nix::float64 elapsed);

  nix::json finish_step(nix::float64 local_push, nix::float64 barrier_wait,
                        MPI_Comm comm = MPI_COMM_WORLD);

private:
  struct ThreadTiming {
    std::array<nix::float64, NumPhases> busy{};
    std::array<nix::float64, NumPhases> max_chunk{};
    std::array<nix::float64, NumPhases> wall{};
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

  static nix::json summarize(const std::vector<nix::float64>& values);
};

#endif
