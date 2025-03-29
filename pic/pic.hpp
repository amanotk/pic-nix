// -*- C++ -*-
#ifndef _PIC_HPP_
#define _PIC_HPP_

#include "nix/nix.hpp"

#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/diag.hpp"

#include "nix/xtensor_halo3d.hpp"
#include "nix/xtensor_packer3d.hpp"
#include "nix/xtensor_particle.hpp"
#include "nix/xtensorall.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
using nix::json;

// particle type
using ParticlePtr  = std::shared_ptr<nix::XtensorParticle>;
using ParticleVec  = std::vector<ParticlePtr>;
using ParticleType = ParticlePtr::element_type;

class PicApplication; // forward declaration of Application type
class PicChunk;       // forward declaration of Chunk type
class PicDiag;        // forward declaration of Diag type
class PicPacker;      // forward declaration of Packer type

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
