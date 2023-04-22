// -*- C++ -*-
#ifndef _EXCHUNK3D_HPP_
#define _EXCHUNK3D_HPP_

#include "nix/buffer.hpp"
#include "nix/chunk3d.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/xtensorall.hpp"

#include "nix/xtensor_particle.hpp"
#include "nix/xtensor_halo3d.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Particle;
using nix::ParticlePtr;
using nix::ParticleVec;

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
/// - set_boundary_field()
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
  using ThisType     = ExChunk3D<Order>;
  using Chunk        = typename BaseChunk3D<Order>::ChunkType;
  using MpiBuffer    = typename Chunk::MpiBuffer;
  using MpiBufferPtr = typename Chunk::MpiBufferPtr;
  using Chunk::dims;
  using Chunk::Lbx;
  using Chunk::Lby;
  using Chunk::Lbz;
  using Chunk::Ubx;
  using Chunk::Uby;
  using Chunk::Ubz;
  using Chunk::delx;
  using Chunk::dely;
  using Chunk::delz;
  using Chunk::xlim;
  using Chunk::ylim;
  using Chunk::zlim;
  using Chunk::mpibufvec;
  using Chunk::load;

  // boundary margin
  static constexpr int Nb = Chunk::boundary_margin;

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

protected:
  float64 field_load; ///< field load per grid point (normalized to particle load)

  int                     Ns; ///< number of particle species
  float64                 cc; ///< speed of light
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  xt::xtensor<float64, 4> uj; ///< current density
  xt::xtensor<float64, 5> um; ///< particle moment
  ParticleVec             up; ///< list of particles

  ///
  /// @brief internal data struct
  ///
  struct InternalData {
    int&                     Lbx;
    int&                     Ubx;
    int&                     Lby;
    int&                     Uby;
    int&                     Lbz;
    int&                     Ubz;
    int&                     Ns;
    float64&                 cc;
    std::vector<float64>&    load;
    xt::xtensor<float64, 4>& uf;
    xt::xtensor<float64, 4>& uj;
    xt::xtensor<float64, 5>& um;
    ParticleVec&             up;
  };

  ///
  /// @brief return internal data struct
  ///
  InternalData get_internal_data()
  {
    return {Lbx, Ubx, Lby, Uby, Lbz, Ubz, Ns, cc, load, uf, uj, um, up};
  }

public:
  ExChunk3D(const int dims[3], int id = 0);

  virtual int pack(void* buffer, int address) override;

  virtual int unpack(void* buffer, int address) override;

  virtual void allocate();

  virtual void reset_load() override;

  virtual void setup(json& config) override;

  virtual void get_energy(float64& efd, float64& bfd, float64 particle[]);

  virtual void get_diverror(float64& efd, float64& bfd);

  virtual void push_efd(float64 delt);

  virtual void push_bfd(float64 delt);

  virtual void push_velocity(float64 delt);

  virtual void push_position(float64 delt);

  virtual void deposit_current(float64 delt);

  virtual void deposit_moment();

  virtual void set_boundary_begin(int mode = 0) override;

  virtual void set_boundary_end(int mode = 0) override;

  template <typename DataPacker>
  int pack_diagnostic(DataPacker packer, uint8_t* buffer, int address)
  {
    return packer(get_internal_data(), buffer, address);
  }
};

// first-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<1> {
  using ChunkType = nix::Chunk3D<2, ParticlePtr>;
};

// second-order shape function requires 2 boundary margins
template <>
struct BaseChunk3D<2> {
  using ChunkType = nix::Chunk3D<2, ParticlePtr>;
};

// third-order shape function requires 3 boundary margins
template <>
struct BaseChunk3D<3> {
  using ChunkType = nix::Chunk3D<3, ParticlePtr>;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
