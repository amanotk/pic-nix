// -*- C++ -*-
#include "ascent.hpp"
#include "../pic_application.hpp"
#include "../pic_chunk.hpp"

#include "../insitu/blueprint_builder.hpp"

#include <filesystem>
#include <string>

void AscentDiag::operator()(nix::json& config)
{
  auto data = interface->get_data();
  if (!require_diagnostic(data.curstep, config)) {
    return;
  }

  const auto actions = config.value("actions", std::string{});
  if (actions.empty()) {
    ERROR << "Ascent diagnostic requires an `actions` path";
    return;
  }

  picnix::insitu::BlueprintOptions options;
  if (config.contains("publish")) {
    if (!config["publish"].is_object()) {
      ERROR << "Ascent diagnostic `publish` must be an object";
      return;
    }
    const auto& publish = config["publish"];
    options.raw         = publish.value("raw", options.raw);
    options.centered    = publish.value("centered", options.centered);
    options.particles   = publish.value("particles", options.particles);
  }

  std::vector<PicChunk*> chunks;
  chunks.reserve(data.chunkvec.size());
  for (auto& chunk : data.chunkvec) {
    chunks.push_back(static_cast<PicChunk*>(chunk.get()));
  }

  const auto actions_path = std::filesystem::path(info->config_dir) / actions;
  auto       publication =
      picnix::insitu::BlueprintBuilder::build(chunks, data.curstep, data.curtime, options);
  runtime.publish_execute(publication.node, actions_path);
}

void AscentDiag::shutdown()
{
  if (shutdown_called) {
    return;
  }

  shutdown_called = true;
  runtime.shutdown();
}
