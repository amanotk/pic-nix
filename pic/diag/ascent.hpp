// -*- C++ -*-
#ifndef _PIC_DIAG_ASCENT_HPP_
#define _PIC_DIAG_ASCENT_HPP_

#include "nix/diag/ascent/runtime.hpp"
#include "pic/pic_application.hpp"
#include "pic/pic_chunk.hpp"
#include "pic/pic_diag.hpp"

///
/// @brief Optional Ascent in-situ diagnostic.
///
class AscentDiag : public PicDiag
{
public:
  AscentDiag(PtrInterface interface) : PicDiag("ascent", std::move(interface))
  {
  }

  void operator()(nix::json& config) override;
  void shutdown() override;

private:
  bool               shutdown_called = false;
  nix::AscentRuntime runtime;
};

#endif
