// -*- C++ -*-
#ifndef _HYBRID_PACKER_HPP_
#define _HYBRID_PACKER_HPP_

#include "hybrid/hybrid.hpp"
#include "hybrid/hybrid_chunk.hpp"

#include "nix/xtensor/xtensor_packer3d.hpp"

namespace hybrid
{
class HybridPacker : public nix::XtensorPacker3D
{
public:
  using chunk_data_type = HybridChunk::DataContainer;

  virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) = 0;

  virtual size_t operator()(HybridChunk* chunk, uint8_t* buffer, int address)
  {
    return operator()(chunk->get_internal_data(), buffer, address);
  }
};
} // namespace hybrid

#endif
