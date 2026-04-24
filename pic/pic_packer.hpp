// -*- C++ -*-
#ifndef _PIC_PACKER_HPP_
#define _PIC_PACKER_HPP_

#include "pic.hpp"

///
/// @brief Packer for 3D PIC Simulations
///
class PicPacker : public nix::XtensorPacker3D
{
public:
  using chunk_data_type = PicChunk::data_type;

  virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) = 0;

  // adapter for generic parallel diag that passes PicChunk*
  virtual size_t operator()(PicChunk* chunk, uint8_t* buffer, int address)
  {
    return operator()(chunk->get_internal_data(), buffer, address);
  }
};

#endif
