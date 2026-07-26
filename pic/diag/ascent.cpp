// -*- C++ -*-
#include "ascent.hpp"
#include "../pic_application.hpp"
#include "../pic_chunk.hpp"

#include "../insitu/blueprint_builder.hpp"

#include <filesystem>
#include <mpi.h>
#include <stdexcept>
#include <string>

void AscentDiag::operator()(nix::json& config)
{
  auto data = interface->get_data();
  if (!require_diagnostic(data.curstep, config)) {
    return;
  }

  if (!config.contains("actions") || !config["actions"].is_string() || config["actions"].empty()) {
    ERROR << "Ascent diagnostic requires a nonempty string `actions` path";
    return;
  }
  const auto actions = config["actions"].get<std::string>();

  picnix::insitu::BlueprintOptions options;
  if (config.contains("publish")) {
    if (!config["publish"].is_object()) {
      ERROR << "Ascent diagnostic `publish` must be an object";
      return;
    }
    const auto& publish = config["publish"];
    for (const auto& key : {"raw", "centered", "particles"}) {
      if (publish.contains(key) && !publish[key].is_boolean()) {
        ERROR << fmt::format("Ascent diagnostic publish option `{}` must be boolean", key);
        return;
      }
    }
    options.raw       = publish.value("raw", options.raw);
    options.centered  = publish.value("centered", options.centered);
    options.particles = publish.value("particles", options.particles);
  }

  std::vector<PicChunk*> chunks;
  chunks.reserve(data.chunkvec.size());
  for (auto& chunk : data.chunkvec) {
    chunks.push_back(static_cast<PicChunk*>(chunk.get()));
  }

  const auto actions_path = std::filesystem::path(info->config_dir) / actions;
  if (!std::filesystem::is_regular_file(actions_path)) {
    ERROR << fmt::format("Ascent actions file does not exist: {}", actions_path.string());
    return;
  }

  try {
    auto publication =
        picnix::insitu::BlueprintBuilder::build(chunks, data.curstep, data.curtime, options);
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
