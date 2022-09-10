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
  using MpiBuffer    = typename Chunk3D<Nb>::MpiBuffer;
  using PtrMpiBuffer = typename Chunk3D<Nb>::PtrMpiBuffer;
  using T_field      = xt::xtensor<float64, 4>;
  using T_moment     = xt::xtensor<float64, 5>;
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
    PackEmf      = 0,
    PackCur      = 1,
    PackMom      = 2,
    PackParticle = 3,
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

  ParticleVec up; ///< list of particles
  T_field     uf; ///< electromagnetic field
  T_field     uj; ///< current density
  T_moment    um; ///< particle moment
  float64     cc; ///< speed of light

public:
  ExChunk3D(const int dims[3], const int id = 0);

  virtual ~ExChunk3D() override;

  virtual int pack(void *buffer, const int address) override;

  virtual int unpack(void *buffer, const int address) override;

  virtual void allocate_memory(const int Np, const int Ns);

  virtual int pack_diagnostic(const int mode, void *buffer, const int address);

  virtual int pack_diagnostic_field(void *buffer, const int address, T_field &u);

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
