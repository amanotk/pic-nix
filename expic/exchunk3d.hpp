// -*- C++ -*-
#ifndef _EXCHUNK3D_HPP_
#define _EXCHUNK3D_HPP_

#ifndef PICNIX_SHAPE_ORDER
#define PICNIX_SHAPE_ORDER 2
#endif

#include "nix/buffer.hpp"
#include "nix/chunk3d.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/xtensorall.hpp"

#include "nix/xtensor_halo3d.hpp"
#include "nix/xtensor_particle.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
using nix::json;
using nix::ParticlePtr;
using nix::ParticleVec;
using ParticleType = nix::ParticlePtr::element_type;

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
class ExChunk3D : public nix::Chunk3D<ParticleType>
{
public:
  using this_type    = ExChunk3D;
  using Chunk        = typename nix::Chunk3D<ParticleType>;
  using MpiBuffer    = typename Chunk::MpiBuffer;
  using MpiBufferPtr = typename Chunk::MpiBufferPtr;
  using Chunk::dims;
  using Chunk::boundary_margin;
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
  using Chunk::option;
  using Chunk::load;

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
  int order;     ///< order of shape function
  int dimension; ///< dimension of the simulation

  int                     Ns; ///< number of particle species
  float64                 cc; ///< speed of light
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  xt::xtensor<float64, 4> uj; ///< current density
  xt::xtensor<float64, 5> um; ///< particle moment
  xt::xtensor<float64, 5> ff; ///< electric field for Friedmann filter
  ParticleVec             up; ///< list of particles

  ///
  /// @brief internal data struct
  ///
  struct InternalData {
    int&     boundary_margin;
    int&     Lbx;
    int&     Ubx;
    int&     Lby;
    int&     Uby;
    int&     Lbz;
    int&     Ubz;
    int&     Ns;
    float64& cc;
    float64& delx;
    float64& dely;
    float64& delz;
    float64 (&xlim)[3];
    float64 (&ylim)[3];
    float64 (&zlim)[3];
    std::vector<float64>&    load;
    xt::xtensor<float64, 4>& uf;
    xt::xtensor<float64, 4>& uj;
    xt::xtensor<float64, 5>& um;
    xt::xtensor<float64, 5>& ff;
    ParticleVec&             up;
    json&                    option;
  };

  ///
  /// @brief return internal data struct
  ///
  InternalData get_internal_data()
  {
    // clang-format off
    return {boundary_margin,
            Lbx,
            Ubx,
            Lby,
            Uby,
            Lbz,
            Ubz,
            Ns,
            cc,
            delx,
            dely,
            delz,
            xlim,
            ylim,
            zlim,
            load,
            uf,
            uj,
            um,
            ff,
            up,
            option};
    // clang-format on
  }

public:
  using data_type = InternalData;

  ExChunk3D(const int dims[3], const bool has_dim[3], int id = 0);

  virtual int64_t get_size_byte() override;

  virtual int pack(void* buffer, int address) override;

  virtual int unpack(void* buffer, int address) override;

  virtual void allocate();

  virtual void reset_load() override;

  virtual void setup(json& config) override;

  virtual void init_friedman();

  virtual bool set_boundary_probe(int mode = 0, bool wait = true) override;

  virtual void set_boundary_pack(int mode = 0) override;

  virtual void set_boundary_unpack(int mode = 0) override;

  virtual void set_boundary_begin(int mode = 0) override;

  virtual void set_boundary_end(int mode = 0) override;

  virtual void get_energy(float64& efd, float64& bfd, float64 particle[]);

  virtual void get_diverror(float64& efd, float64& bfd);

  virtual void sort_particle(ParticleVec& particle) override;

  virtual void count_particle(ParticlePtr particle, int Lbp, int Ubp, bool reset = true) override;

  virtual void push_position(float64 delt);

  virtual void push_velocity(float64 delt);

  virtual void deposit_current(float64 delt);

  virtual void deposit_moment();

  virtual void push_efd(float64 delt);

  virtual void push_bfd(float64 delt);

  template <typename DataPacker>
  size_t pack_diagnostic(DataPacker packer, uint8_t* buffer, int address)
  {
    return packer(get_internal_data(), buffer, address);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
