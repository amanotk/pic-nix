// -*- C++ -*-
#ifndef _PIC_CHUNK_WRITER_HPP_
#define _PIC_CHUNK_WRITER_HPP_

#include "nix/diag/chunk_writer.hpp"
#include "pic_diag.hpp"
#include "pic_packer.hpp"

// PIC alias for generic chunk writer
using PicChunkDiagWriter = nix::ChunkDiagWriter<PicDiag, PicPacker>;

#endif
