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
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
