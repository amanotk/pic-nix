// -*- C++ -*-
#ifndef _EXCHUNK3D_HPP_
#define _EXCHUNK3D_HPP_

#include "nix/buffer.hpp"
#include "nix/chunk3d.hpp"
#include "nix/debug.hpp"
#include "nix/jsonio.hpp"
#include "nix/nix.hpp"
#include "nix/particle.hpp"
#include "nix/xtensorall.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Particle;
using nix::ParticleVec;
using nix::PtrParticle;

// trick to map from order of shape function to number of boundary margins
template <int Order>
struct BaseChunk3D;

///
/// @brief Chunk for 3D Explicit PIC Simulations
/// @tparam Order order of shape function
///
/// This class implements the standard explicit FDTD PIC simulation scheme. The order of shape
/// function is given as a template parameter, so that specific implementation should be provided
/// through explicit instantiation.
///
/// Problem specific codes should be provided by making a derived class which implements the
/// following virtual methods (if the default ones are not appropriate):
///
/// - setup()
///   physics-wise initial condition (both for particles and fields)
/// - set_boundary_physical()
///   non-periodic boundary condition for fields
/// - set_boundary_particle()
///   non-periodic boundary condition for particles
/// - inject_particle()
///   particle injection into the system
///
/// In addition, custom diagnostics routines may also be implemented depending on the needs of
/// applications.
///
template <int Order>
class ExChunk3D : public BaseChunk3D<Order>::ChunkType
{
public:
  using Chunk        = typename BaseChunk3D<Order>::ChunkType;
  using MpiBuffer    = typename Chunk::MpiBuffer;
  using PtrMpiBuffer = typename Chunk::PtrMpiBuffer;
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
  using Chunk::delx;
  using Chunk::dely;
  using Chunk::delz;
  using Chunk::xlim;
  using Chunk::ylim;
  using Chunk::zlim;
  using Chunk::mpibufvec;

  // boundary margin
  static constexpr int Nb = Chunk::boundary_margin;

  // mode for diagnostic
  enum DiagnosticMode {
    DiagnosticLoad     = 0,
    DiagnosticX        = 1,
    DiagnosticY        = 2,
    DiagnosticZ        = 3,
    DiagnosticEmf      = 4,
    DiagnosticCur      = 5,
    DiagnosticMom      = 6,
    DiagnosticParticle = 10,
    DiagnosticCustom   = 20,
  };

  // mode for load
  enum LoadMode {
    LoadEmf      = 0,
    LoadCur      = 1,
    LoadParticle = 2,
    NumLoadMode  = 3, // number of mode
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
  int                     Ns; ///< number of particle species
  float64                 cc; ///< speed of light
  ParticleVec             up; ///< list of particles
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  xt::xtensor<float64, 4> uj; ///< current density
  xt::xtensor<float64, 5> um; ///< particle moment

public:
  ExChunk3D(const int dims[3], const int id = 0);

  virtual int pack(void* buffer, const int address) override;

  virtual int unpack(void* buffer, const int address) override;

  virtual void allocate();

  virtual int pack_diagnostic(const int mode, void* buffer, const int address);

  virtual void setup(json& config) override;

  virtual void get_energy(float64& efd, float64& bfd, float64 particle[]);

  virtual void get_diverror(float64& efd, float64& bfd);

  virtual void push_efd(const float64 delt);

  virtual void push_bfd(const float64 delt);

  virtual void push_velocity(const float64 delt);

  virtual void push_position(const float64 delt);

  virtual void deposit_current(const float64 delt);

  virtual void deposit_moment();

  virtual void set_boundary_begin(const int mode = 0) override;

  virtual void set_boundary_end(const int mode = 0) override;
};

// first-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<1> {
  using ChunkType = nix::Chunk3D<2>;
};

// second-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<2> {
  using ChunkType = nix::Chunk3D<2>;
};

// third-order shape function requires 3 boundary margins
template <>
struct BaseChunk3D<3> {
  using ChunkType = nix::Chunk3D<3>;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
