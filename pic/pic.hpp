// -*- C++ -*-
#ifndef _PIC_HPP_
#define _PIC_HPP_

#include "nix/nix.hpp"

#include "nix/buffer.hpp"
#include "nix/debug.hpp"

#include "nix/xtensorall.hpp"
#include "nix/xtensor_halo3d.hpp"
#include "nix/xtensor_particle.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
using nix::json;
using nix::ParticlePtr;
using nix::ParticleVec;
using ParticleType = nix::ParticlePtr::element_type;

class PicChunk;       // forward declaration of Chunk type
class PicApplication; // forward declaration of Application type

// mode for load
enum LoadMode {
  LoadField    = 0,
  LoadParticle = 1,
  NumLoadMode  = 2, // number of mode
};

// mode for boundary exchange
enum BoundaryMode {
  BoundaryEmf      = 0,
  BoundaryCur      = 1,
  BoundaryMom      = 2,
  BoundaryParticle = 3,
  NumBoundaryMode  = 4, // number of mode
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
