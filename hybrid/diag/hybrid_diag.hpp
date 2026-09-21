// -*- C++ -*-
#ifndef _HYBRID_BASE_DIAG_HPP_
#define _HYBRID_BASE_DIAG_HPP_

#include "hybrid/hybrid.hpp"

#include "nix/diag.hpp"
#include "nix/nixio.hpp"

namespace hybrid
{
using nix::json;
using nix::float64;
using nix::typedefs::MPI_FLOAT64_T;

class HybridDiag : public nix::Diag
{
public:
  using base_type = nix::Diag;
  using base_type::base_type; // inherit constructors
  using app_type     = HybridApplication;
  using data_type    = app_type::data_type;
  using chunk_type   = HybridChunk;
  using PtrInterface = std::shared_ptr<HybridApplicationInterface>;

protected:
  PtrInterface interface; ///< interface

public:
  HybridDiag(std::string name, PtrInterface interface) : nix::Diag(name), interface(interface)
  {
    make_sure_directory_exists(format_dirname(""));
  }
};
} // namespace hybrid

#endif
