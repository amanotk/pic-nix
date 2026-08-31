// -*- C++ -*-
#ifndef _HYBRID_CHUNK_HPP_
#define _HYBRID_CHUNK_HPP_

#include "hybrid.hpp"

#include "nix/array_types.hpp"
#include "nix/chunk.hpp"
#include "nix/xtensor/xtensor_particle.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace hybrid
{
class HybridChunk : public nix::Chunk
{
public:
  struct DataContainer;
  using base_type    = nix::Chunk;
  using MpiBuffer    = base_type::MpiBuffer;
  using MpiBufferPtr = base_type::MpiBufferPtr;
  using ParticlePtr  = std::shared_ptr<nix::XtensorParticle>;
  using ParticleVec  = std::vector<ParticlePtr>;

  struct ParticleDisplacement {
    nix::float64                ratio    = 0;
    int                         species  = -1;
    int                         particle = -1;
    int64_t                     id       = 0;
    std::array<nix::float64, 3> before   = {};
    std::array<nix::float64, 3> after    = {};
  };

  struct DataContainer {
    int&                        boundary_margin;
    int&                        Lbx;
    int&                        Ubx;
    int&                        Lby;
    int&                        Uby;
    int&                        Lbz;
    int&                        Ubz;
    int&                        order;
    int&                        num_species;
    nix::float64&               light_speed;
    nix::float64&               adiabatic_index;
    std::vector<nix::float64>&  load;
    nix::json&                  option;
    nix::Array3D<nix::float64>& eta;
    nix::Array3D<nix::float64>& chi;
    nix::Array4D<nix::float64>& fluid;
    nix::Array4D<nix::float64>& field_cell;
    nix::Array4D<nix::float64>& field_staggered;
    nix::Array5D<nix::float64>& moment_kinetic;
    nix::Array4D<nix::float64>& current_kinetic;
    nix::Array4D<nix::float64>& background_cell;
    nix::Array4D<nix::float64>& background_x_face;
    nix::Array4D<nix::float64>& background_y_face;
    nix::Array4D<nix::float64>& background_z_face;
    ParticleVec&                particles;
    nix::Array5D<nix::float64>& phase_cell;
    nix::Array5D<nix::float64>& phase_face;
    nix::Array4D<nix::float64>& curl_b;
    nix::Array4D<nix::float64>& ohm_moment;
    nix::Array4D<nix::float64>& resistive_field;
    nix::Array5D<nix::float64>& fluid_flux;
    nix::Array4D<nix::float64>& field_flux;
    nix::Array4D<nix::float64>& work_fluid;
    nix::Array4D<nix::float64>& work_field_cell;
    nix::Array4D<nix::float64>& work_field_staggered;
    nix::Array4D<nix::float64>& solver_left;
    nix::Array4D<nix::float64>& solver_right;
    nix::Array4D<nix::float64>& solver_field_x;
    nix::Array4D<nix::float64>& solver_field_y;
    nix::Array4D<nix::float64>& solver_field_z;
    nix::Array4D<nix::float64>& ohm_source;
    nix::Array3D<nix::float64>& filter_scratch;
  };

protected:
  int          order;
  int          num_species;
  nix::float64 light_speed;
  nix::float64 adiabatic_index;

  nix::Array3D<nix::float64> eta;
  nix::Array3D<nix::float64> chi;
  nix::Array4D<nix::float64> fluid;
  nix::Array4D<nix::float64> field_cell;
  nix::Array4D<nix::float64> field_staggered;
  nix::Array5D<nix::float64> moment_kinetic;
  nix::Array4D<nix::float64> current_kinetic;
  nix::Array4D<nix::float64> background_cell;
  nix::Array4D<nix::float64> background_x_face;
  nix::Array4D<nix::float64> background_y_face;
  nix::Array4D<nix::float64> background_z_face;
  ParticleVec                particles;

  nix::Array5D<nix::float64> phase_cell;
  nix::Array5D<nix::float64> phase_face;
  nix::Array4D<nix::float64> curl_b;
  nix::Array4D<nix::float64> ohm_moment;
  nix::Array4D<nix::float64> resistive_field;

  nix::Array5D<nix::float64> fluid_flux;
  nix::Array4D<nix::float64> field_flux;
  nix::Array4D<nix::float64> work_fluid;
  nix::Array4D<nix::float64> work_field_cell;
  nix::Array4D<nix::float64> work_field_staggered;
  nix::Array4D<nix::float64> solver_left;
  nix::Array4D<nix::float64> solver_right;
  nix::Array4D<nix::float64> solver_field_x;
  nix::Array4D<nix::float64> solver_field_y;
  nix::Array4D<nix::float64> solver_field_z;
  nix::Array4D<nix::float64> ohm_source;
  nix::Array3D<nix::float64> filter_scratch;

public:
  HybridChunk(nix::Dims3D dims, nix::Bool3D has_dim, int id = 0);

  ~HybridChunk() override = default;

  DataContainer get_internal_data();

  int get_order() const;

  int get_num_species() const;

  int64_t get_size_byte() const override;

  int pack(void* buffer, int address) override;

  int unpack(void* buffer, int address) override;

  void allocate();

  void allocate_mpi_buffers();

  bool exchanges_idle() const;

  void boundary_pack(nix::Array4D<nix::float64>& array, BoundaryMode mode);

  void boundary_unpack(nix::Array4D<nix::float64>& array, BoundaryMode mode);

  void boundary_begin(nix::Array4D<nix::float64>& array, BoundaryMode mode);

  void boundary_end(nix::Array4D<nix::float64>& array, BoundaryMode mode);

  void boundary_pack(nix::Array5D<nix::float64>& array, BoundaryMode mode);

  void boundary_unpack(nix::Array5D<nix::float64>& array, BoundaryMode mode);

  void boundary_begin(nix::Array5D<nix::float64>& array, BoundaryMode mode);

  void boundary_end(nix::Array5D<nix::float64>& array, BoundaryMode mode);

  void particle_boundary_pack();

  void particle_boundary_unpack();

  void particle_boundary_begin();

  void particle_boundary_end();

  bool particle_boundary_probe(bool wait = true);

  ParticleDisplacement get_max_particle_displacement() const;

  void prepare_particle_migration();

  void reset_load() override;

  void setup(nix::json& config) override;
};
} // namespace hybrid

#endif
