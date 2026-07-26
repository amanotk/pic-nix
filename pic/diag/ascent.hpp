// -*- C++ -*-
#ifndef _PIC_DIAG_ASCENT_HPP_
#define _PIC_DIAG_ASCENT_HPP_

#include "../insitu/ascent_runtime.hpp"
#include "../pic_application.hpp"
#include "../pic_chunk.hpp"
#include "../pic_diag.hpp"

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
  bool                          shutdown_called = false;
  picnix::insitu::AscentRuntime runtime;
};

#endif
