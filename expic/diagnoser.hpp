// -*- C++ -*-
#ifndef _DIAGNOSER_HPP_
#define _DIAGNOSER_HPP_

#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/nixio.hpp"
#include "nix/random.hpp"

#include "diag/base.hpp"
#include "diag/field.hpp"
#include "diag/history.hpp"
#include "diag/load.hpp"
#include "diag/particle.hpp"
#include "diag/resource.hpp"
#include "diag/tracer.hpp"

///
/// @brief Diagnoser
///
template <typename App, typename Data>
class Diagnoser
{
protected:
  using PtrDiagnostic = std::unique_ptr<BaseDiag<App, Data>>;

  std::shared_ptr<DiagInfo>  info;
  std::vector<PtrDiagnostic> diagnostics;

public:
  Diagnoser(std::string basedir, std::string iomode)
  {
    info = std::make_shared<DiagInfo>(basedir, iomode);

    diagnostics.push_back(std::make_unique<HistoryDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<ResourceDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<LoadDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<FieldDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<ParticleDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<PickupTracerDiag<App, Data>>(info));
    diagnostics.push_back(std::make_unique<TracerDiag<App, Data>>(info));
  }

  void diagnose(json& config, App& app, Data& data)
  {
    if (config.contains("name") == false)
      return;

    for (auto& diag : diagnostics) {
      if (diag->match(config["name"])) {
        (*diag)(config, app, data);
        break;
      }
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
