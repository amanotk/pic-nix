// -*- C++ -*-
#include "ascent.hpp"
#include "pic/pic_application.hpp"
#include "pic/pic_chunk.hpp"

#include "pic/diag/ascent/blueprint_builder.hpp"

#include <filesystem>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <system_error>

void AscentDiag::operator()(nix::json& config)
{
  auto data = interface->get_data();
  if (config.contains("interval") &&
      (!config["interval"].is_number_integer() || config["interval"].get<int64_t>() <= 0)) {
    ERROR << "Ascent diagnostic `interval` must be a positive integer";
    return;
  }
  if (!require_diagnostic(data.curstep, config)) {
    return;
  }

  if (!config.contains("actions") || !config["actions"].is_string() || config["actions"].empty()) {
    ERROR << "Ascent diagnostic requires a nonempty string `actions` path";
    return;
  }
  const auto actions = config["actions"].get<std::string>();

  pic_ascent::BlueprintOptions options;
  for (const auto key :
       {"publish_electric_field", "publish_magnetic_field", "publish_mass_current",
        "publish_energy_momentum", "publish_raw_fields", "publish_raw_particles"}) {
    if (config.contains(key) && !config[key].is_boolean()) {
      ERROR << fmt::format("Ascent diagnostic `{}` must be a boolean", key);
      return;
    }
  }
  options.electric_field  = config.value("publish_electric_field", options.electric_field);
  options.magnetic_field  = config.value("publish_magnetic_field", options.magnetic_field);
  options.mass_current    = config.value("publish_mass_current", options.mass_current);
  options.energy_momentum = config.value("publish_energy_momentum", options.energy_momentum);
  options.raw_fields      = config.value("publish_raw_fields", options.raw_fields);
  options.raw_particles   = config.value("publish_raw_particles", options.raw_particles);

  std::vector<PicChunk*> chunks;
  chunks.reserve(data.chunkvec.size());
  for (auto& chunk : data.chunkvec) {
    chunks.push_back(static_cast<PicChunk*>(chunk.get()));
  }

  const auto      actions_path = std::filesystem::path(info->config_dir) / actions;
  std::error_code path_error;
  const int       local_actions_valid =
      std::filesystem::is_regular_file(actions_path, path_error) ? 1 : 0;
  int global_actions_valid = 0;
  MPI_Allreduce(&local_actions_valid, &global_actions_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (global_actions_valid == 0) {
    if (local_actions_valid == 0) {
      ERROR << fmt::format("Ascent actions file is not readable on rank {}: {}", info->world_rank,
                           actions_path.string());
    }
    return;
  }

  if (options.mass_current || options.energy_momentum) {
    interface->calculate_moment();
  }

  try {
    auto publication = pic_ascent::BlueprintBuilder::build(chunks, data.curstep, data.curtime,
                                                           interface->get_configuration(), options);
    runtime.publish_execute(publication.node, actions_path);
  } catch (const std::exception& error) {
    ERROR << fmt::format("Ascent diagnostic failed on rank {} for `{}`: {}", info->world_rank,
                         actions_path.string(), error.what());
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

void AscentDiag::shutdown()
{
  if (shutdown_called) {
    return;
  }

  shutdown_called = true;
  runtime.shutdown();
}
