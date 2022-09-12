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

// trick to map from order of shape function to number of boundary margins
template <int Order>
struct BaseChunk3D;

///
template <int Order>
class ExChunk3D : public BaseChunk3D<Order>::ChunkType
{
public:
  using json         = common::json;
  using T_field      = xt::xtensor<float64, 4>;
  using T_moment     = xt::xtensor<float64, 5>;
  using Chunk        = typename BaseChunk3D<Order>::ChunkType;
  using MpiBuffer    = typename Chunk::MpiBuffer;
  using PtrMpiBuffer = typename Chunk::PtrMpiBuffer;
  using Chunk::boundary_margin;
  using Chunk::dims;
  using Chunk::Lbx;
  using Chunk::Lby;
  using Chunk::Lbz;
  using Chunk::Ubx;
  using Chunk::Uby;
  using Chunk::Ubz;
  using Chunk::xc;
  using Chunk::yc;
  using Chunk::zc;
  using Chunk::delh;
  using Chunk::xlim;
  using Chunk::ylim;
  using Chunk::zlim;
  using Chunk::RecvMode;
  using Chunk::SendMode;
  using Chunk::mpibufvec;

  // boundary margin
  static constexpr int Nb = boundary_margin;

  // mode for diagnostic
  enum DiagnosticMode {
    DiagnosticEmf      = 0,
    DiagnosticCur      = 1,
    DiagnosticMom      = 2,
    DiagnosticParticle = 3,
  };

  // mode for boundary exchange
  enum BoundaryMode {
    BoundaryEmf      = 0,
    BoundaryCur      = 1,
    BoundaryMom      = 2,
    BoundaryParticle = 3,
    NumBoundaryMode  = 4, // number of mode
  };

protected:
  int         Ns; ///< number of particle species
  float64     cc; ///< speed of light
  ParticleVec up; ///< list of particles
  T_field     uf; ///< electromagnetic field
  T_field     uj; ///< current density
  T_moment    um; ///< particle moment

public:
  ExChunk3D(const int dims[3], const int id = 0);

  virtual ~ExChunk3D() override;

  virtual int pack(void *buffer, const int address) override;

  virtual int unpack(void *buffer, const int address) override;

  virtual void allocate();

  virtual int pack_diagnostic(const int mode, void *buffer, const int address);

  virtual int pack_diagnostic_field(void *buffer, const int address, T_field &u);

  virtual void setup(json &config) override;

  virtual void push_efd(const float64 delt);

  virtual void push_mfd(const float64 delt);

  virtual void push_velocity(const float64 delt);

  virtual void push_position(const float64 delt);

  virtual void deposit_current(const float64 delt);

  virtual void set_boundary_begin(const int mode = 0) override;

  virtual void set_boundary_end(const int mode = 0) override;
};

// first-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<1> {
  using ChunkType = Chunk3D<2>;
};

// second-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<2> {
  using ChunkType = Chunk3D<2>;
};

// third-order shape function requires 3 boundary margins
template <>
struct BaseChunk3D<3> {
  using ChunkType = Chunk3D<3>;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
