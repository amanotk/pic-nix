// -*- C++ -*-
#ifndef _PIC_DIAG_HPP_
#define _PIC_DIAG_HPP_

#include "nix/nixio.hpp"
#include "pic.hpp"

using namespace nix::typedefs;
using namespace nixio;

///
/// @brief Diagnostic for 3D PIC Simulations
///
class PicDiag : public nix::Diag
{
public:
  using base_type = nix::Diag;
  using base_type::base_type; // inherit constructors
  using app_type     = PicApplication;
  using data_type    = app_type::data_type;
  using chunk_type   = PicChunk;
  using PtrInterface = std::shared_ptr<PicApplicationInterface>;

protected:
  PtrInterface interface; ///< interface

public:
  // constructor
  PicDiag(std::string name, PtrInterface interface) : nix::Diag(name), interface(interface)
  {
    make_sure_directory_exists(format_dirname(""));
  }
};

#endif
