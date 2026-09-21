// -*- C++ -*-
#ifndef _HYBRID_CHUNK_WRITER_HPP_
#define _HYBRID_CHUNK_WRITER_HPP_

#include "hybrid/diag/hybrid_diag.hpp"
#include "hybrid/hybrid_packer.hpp"

#include "nix/diag/chunk_writer.hpp"

namespace hybrid
{
using HybridChunkDiagWriter = nix::ChunkDiagWriter<HybridDiag, HybridPacker>;
} // namespace hybrid

#endif
