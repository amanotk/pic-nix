// -*- C++ -*-
#ifndef _PIC_CHUNK_HPP_
#define _PIC_CHUNK_HPP_

#include "pic.hpp"

#include "nix/chunk.hpp"

///
/// @brief Chunk for 3D PIC Simulations
///
class PicChunk : public nix::Chunk
{
public:
  struct DataContainer; // forward declaration
  using this_type    = PicChunk;
  using base_type    = nix::Chunk;
  using data_type    = DataContainer;
  using MpiBuffer    = base_type::MpiBuffer;
  using MpiBufferPtr = base_type::MpiBufferPtr;

  /// @brief internal data struct
  struct DataContainer {
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

  /// @brief return internal data struct
  DataContainer get_internal_data()
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

public:
  PicChunk(const int dims[3], const bool has_dim[3], int id = 0);

  virtual ~PicChunk() override = default;

  virtual int64_t get_size_byte() const override;

  virtual int get_order() const;

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

  virtual void sort_particle(ParticleVec& particle);

  virtual void set_boundary_field(int mode);

  virtual void set_boundary_particle(ParticleVec& particle);

  virtual void inject_particle(ParticleVec& particle);

  virtual void push_position(float64 delt);

  virtual void push_velocity(float64 delt);

  virtual void deposit_current(float64 delt);

  virtual void deposit_moment();

  virtual void push_efd(float64 delt);

  virtual void push_bfd(float64 delt);
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
