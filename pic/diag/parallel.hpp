// -*- C++ -*-
#ifndef _PARALLEL_DIAG_HPP_
#define _PARALLEL_DIAG_HPP_

#include "nix/diag/parallel.hpp"
#include "pic_diag.hpp"
#include "pic_packer.hpp"

// Compatibility alias for PIC
using PicParallelDiag = nix::ParallelDiag<PicDiag, PicPacker>;

// Keep ParallelDiag name for backward compatibility
using ParallelDiag = PicParallelDiag;

#endif
