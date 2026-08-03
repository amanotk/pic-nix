// -*- C++ -*-
#include "pic_performance.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace
{
constexpr std::array<const char*, PicPerformance::NumPhases> phase_names = {
    "advance", "current_field", "particle_probe", "particle_exchange", "field_exchange"};

constexpr int push_metric_count  = 2;
constexpr int phase_metric_count = 3;
constexpr int metric_count = push_metric_count + PicPerformance::NumPhases * phase_metric_count;
} // namespace

bool PicPerformance::configure(const nix::json& application)
{
  enabled = false;

  if (application.contains("performance") == false) {
    return true;
  }

  const auto& config = application["performance"];
  if (config.is_object() == false) {
    return false;
  }

  if ((config.contains("enabled") && config["enabled"].is_boolean() == false) ||
      (config.contains("interval") && config["interval"].is_number_integer() == false) ||
      (config.contains("offset") && config["offset"].is_number_integer() == false)) {
    return false;
  }

  enabled  = config.value("enabled", true);
  interval = config.value("interval", 100);
  offset   = config.value("offset", 0);

  if (interval <= 0 || offset < 0 || offset >= interval) {
    return false;
  }

  return true;
}

bool PicPerformance::begin_step(int step, int nthread)
{
  sampling = enabled && step >= offset && (step - offset) % interval == 0;
  if (sampling == false) {
    return false;
  }

  thread_timing.assign(nthread, {});
  parallel_threads = nthread;
  return true;
}

void PicPerformance::set_parallel_threads(int nthread)
{
  assert(nthread > 0 && nthread <= static_cast<int>(thread_timing.size()));
  parallel_threads = nthread;
}

void PicPerformance::record_chunk(Phase phase, nix::float64 elapsed)
{
  if (sampling == false) {
    return;
  }

  int thread = 0;
#ifdef _OPENMP
  thread = omp_get_thread_num();
#endif

  int index = phase_index(phase);
  thread_timing[thread].busy[index] += elapsed;
  thread_timing[thread].max_chunk[index] =
      std::max(thread_timing[thread].max_chunk[index], elapsed);
}

void PicPerformance::record_phase_wall(Phase phase, nix::float64 elapsed)
{
  if (sampling == false) {
    return;
  }

  int thread = 0;
#ifdef _OPENMP
  thread = omp_get_thread_num();
#endif

  thread_timing[thread].wall[phase_index(phase)] = elapsed;
}

nix::json PicPerformance::summarize(const std::vector<nix::float64>& values)
{
  std::vector<nix::float64> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  const auto percentile = [&](nix::float64 q) {
    nix::float64 position = q * static_cast<nix::float64>(sorted.size() - 1);
    size_t       lower    = static_cast<size_t>(std::floor(position));
    size_t       upper    = static_cast<size_t>(std::ceil(position));
    nix::float64 weight   = position - static_cast<nix::float64>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
  };

  auto min_it = std::min_element(values.begin(), values.end());
  auto max_it = std::max_element(values.begin(), values.end());

  return {
      {"size", values.size()},
      {"min", *min_it},
      {"max", *max_it},
      {"mean", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
      {"median", percentile(0.5)},
      {"p95", percentile(0.95)},
      {"min_rank", std::distance(values.begin(), min_it)},
      {"max_rank", std::distance(values.begin(), max_it)},
  };
}

nix::json PicPerformance::finish_step(nix::float64 local_push, nix::float64 barrier_wait,
                                      MPI_Comm comm)
{
  if (sampling == false) {
    return {};
  }

  std::array<nix::float64, metric_count> local{};
  local[0] = local_push;
  local[1] = barrier_wait;

  for (int phase = 0; phase < NumPhases; phase++) {
    nix::float64 wall      = 0.0;
    nix::float64 busy      = 0.0;
    nix::float64 max_chunk = 0.0;

    for (int thread = 0; thread < parallel_threads; thread++) {
      const auto& timing = thread_timing[thread];
      wall               = std::max(wall, timing.wall[phase]);
      busy += timing.busy[phase];
      max_chunk = std::max(max_chunk, timing.max_chunk[phase]);
    }

    int base        = push_metric_count + phase * phase_metric_count;
    local[base + 0] = wall;
    local[base + 1] = wall > 0.0 ? busy / (parallel_threads * wall) : 0.0;
    local[base + 2] = max_chunk;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<nix::float64> gathered;
  if (rank == 0) {
    gathered.resize(size * metric_count);
  }

  MPI_Gather(local.data(), metric_count, MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
             metric_count, MPI_DOUBLE, 0, comm);

  sampling = false;
  if (rank != 0) {
    return {};
  }

  const auto metric_values = [&](int metric) {
    std::vector<nix::float64> values(size);
    for (int process = 0; process < size; process++) {
      values[process] = gathered[process * metric_count + metric];
    }
    return values;
  };

  nix::json result = {
      {"schema_version", 1},
      {"push",
       {
           {"local", summarize(metric_values(0))},
           {"barrier", summarize(metric_values(1))},
       }},
      {"phase", nix::json::object()},
  };

  for (int phase = 0; phase < NumPhases; phase++) {
    int base                            = push_metric_count + phase * phase_metric_count;
    result["phase"][phase_names[phase]] = {
        {"wall", summarize(metric_values(base + 0))},
        {"omp_efficiency", summarize(metric_values(base + 1))},
        {"max_chunk", summarize(metric_values(base + 2))},
    };
  }

  return result;
}
