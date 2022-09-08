// -*- C++ -*-
#ifndef _EXCHUNK3D_HPP_
#define _EXCHUNK3D_HPP_

#include "nix/buffer.hpp"
#include "nix/chunk3d.hpp"
#include "nix/common.hpp"
#include "nix/debug.hpp"
#include "nix/jsonio.hpp"
#include "nix/particle.hpp"
#include "nix/xtensorall.hpp"

///
template <int Nb>
class ExChunk3D : public Chunk3D<Nb>
{
public:
  using Chunk        = Chunk3D<Nb>;
  using PtrMpiBuffer = typename Chunk3D<Nb>::PtrMpiBuffer;
  using Chunk::dims;
  using Chunk::Lbx;
  using Chunk::Lby;
  using Chunk::Lbz;
  using Chunk::Ubx;
  using Chunk::Uby;
  using Chunk::Ubz;
  using Chunk::RecvMode;
  using Chunk::SendMode;
  using Chunk::mpibufvec;

  enum PackMode {
    PackAllQuery = 0,
    PackAll      = 1,
    PackEmfQuery = 2,
    PackEmf      = 3,
    PackMomQuery = 4,
    PackMom      = 5,
  };

  enum BoundaryMode {
    BoundaryEmf      = 0,
    BoundaryCur      = 1,
    BoundaryMom      = 2,
    BoundaryParticle = 3,
    NumBoundaryMode  = 4,
  };

protected:
  int Ns; ///< number of particle species

  ParticleVec             up; ///< list of particles
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  xt::xtensor<float64, 4> uj; ///< current density
  xt::xtensor<float64, 5> um; ///< particle moment
  float64                 cc; ///< speed of light

public:
  ExChunk3D(const int dims[3], const int id = 0);

  virtual ~ExChunk3D() override;

  virtual int pack(const int mode, void *buffer) override;

  virtual int unpack(const int mode, void *buffer) override;

  virtual void allocate_memory(const int Np, const int Ns);

  virtual int pack_diagnostic(const int mode, void *buffer);

  virtual int pack_diagnostic_emf(void *buffer, const bool query);

  virtual int pack_diagnostic_mom(void *buffer, const bool query);

  virtual void setup(const float64 cc, const float64 delh, const int offset[3]);

  virtual void push_efd(const float64 delt);

  virtual void push_mfd(const float64 delt);

  virtual void push_velocity(const float64 delt);

  virtual void push_position(const float64 delt);

  virtual void deposit_current(const float64 delt);

  virtual void set_boundary_begin(const int mode = 0) override;

  virtual void set_boundary_end(const int mode = 0) override;

  virtual void push(const float64 delt) override
  {
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
