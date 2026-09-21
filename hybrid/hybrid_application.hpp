// -*- C++ -*-
#ifndef _HYBRID_APPLICATION_HPP_
#define _HYBRID_APPLICATION_HPP_

#include "hybrid.hpp"

#include "engine/pcc2.hpp"
#include "nix/application.hpp"
#include "nix/chunk.hpp"
#include "nix/diag.hpp"

#include <array>

namespace hybrid
{
class HybridApplicationInterface : public nix::Application::Interface
{
public:
  PtrChunk create_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id) override;

  virtual int get_num_species();
};

class HybridApplication : public nix::Application
{
public:
  using base_type  = nix::Application;
  using MpiCommVec = nix::FixedArray4D<MPI_Comm, NumBoundaryModes, 3, 3, 3>;

  HybridApplication(int argc, char** argv, PtrInterface interface);

  ~HybridApplication() override = default;

  int get_num_species() const;

protected:
  MpiCommVec mpicommvec;
  bool       communicators_initialized = false;
  int        num_species_              = 0;

  void initialize(int argc, char** argv) override;

  void initialize_diagnostic() override;

  void finalize() override;

  void setup_chunks() override;

  bool rebalance() override;

  void set_chunk_communicators();

  void restore_accepted_halos();

  void migrate_particles();

  void update_kinetic_moments();

  void require_kinetic_particles() const;

  bool is_push_needed() override;

  virtual void pcc2_stage_completed(engine::Pcc2Stage stage);

  void push() override;
};
} // namespace hybrid

#endif
