// -*- C++ -*-
#ifndef _PIC_APPLICATION_HPP_
#define _PIC_APPLICATION_HPP_

#include "diagnoser.hpp"
#include "exchunk3d.hpp"
#include "nix/application.hpp"
#include "nix/chunkmap.hpp"

#include <taskflow/taskflow.hpp>

using namespace nix::typedefs;
using nix::json;
using nix::Balancer;
using nix::Logger;

///
/// @brief Application for 3D Explicit PIC Simulations
///
/// This class takes care of initialization of various objects (ChunkMap, Balancer, etc.) and a main
/// loop for explicit PIC simulations. Core numerical solvers are implemented in a chunk class, and
/// this class only deals with the management of a collection of chunks.
///
/// A problem specific application class must be defined by deriving this class, which should
/// override the virtual factory method create_chunk() to return a pointer to a problem specific
/// chunk instance. In addition, custom diagnostics routines may also be implemented through the
/// virtual method diagnostic().
///
template <typename Chunk>
class PicApplication : public nix::Application<Chunk, nix::ChunkMap<3>>
{
public:
  using this_type     = PicApplication<Chunk>;
  using base_type     = nix::Application<Chunk, nix::ChunkMap<3>>;
  using chunk_type    = Chunk;
  using DiagnoserType = Diagnoser<this_type, typename base_type::InternalData>;
  using PtrDiagnoser  = std::unique_ptr<DiagnoserType>;
  using MpiCommVec    = xt::xtensor_fixed<MPI_Comm, xt::xshape<Chunk::NumBoundaryMode, 3, 3, 3>>;

protected:
  using base_type::cfgparser;
  using base_type::argparser;
  using base_type::balancer;
  using base_type::logger;
  using base_type::chunkvec;
  using base_type::chunkmap;
  using base_type::ndims;
  using base_type::cdims;
  using base_type::curstep;
  using base_type::curtime;
  using base_type::nprocess;
  using base_type::thisrank;

  int          Ns;         ///< number of species
  int          momstep;    ///< step at which moment quantities are cached
  MpiCommVec   mpicommvec; ///< MPI Communicators
  PtrDiagnoser diagnoser;  ///< diagnostic handler

  virtual void initialize(int argc, char** argv) override;

  virtual void set_chunk_communicator();

  virtual void setup_chunks() override;

  virtual bool rebalance() override;

  virtual void finalize() override;

  virtual std::unique_ptr<Chunk> create_chunk(const int dims[], const bool has_dim[],
                                              int id) override = 0;

  virtual std::unique_ptr<DiagnoserType> create_diagnoser()
  {
    return std::make_unique<DiagnoserType>(this->get_basedir(), this->get_iomode(), this->Ns);
  }

  virtual std::string get_basedir() override
  {
    auto tmpdir  = std::getenv("PICNIX_TMPDIR");
    auto basedir = cfgparser->get_application().value("basedir", "");

    if (tmpdir == nullptr) {
      return basedir;
    } else {
      return std::filesystem::path(tmpdir) / basedir;
    }
  }

  virtual std::string get_iomode()
  {
    return cfgparser->get_application().value("iomode", "mpiio");
  }

public:
  PicApplication(int argc, char** argv);

  virtual json to_json() override;

  virtual bool from_json(json& state) override;

  virtual void diagnostic() override;

  virtual void push_taskflow();

  virtual void calculate_moment_taskflow();

  virtual void push_openmp();

  virtual void calculate_moment_openmp();

  virtual void push() override
  {
    DEBUG2 << "push() start";
    float64 wclock1 = nix::wall_clock();

    auto option = cfgparser->get_application()["option"];

    if (option.contains("thread") == false || option["thread"] == "openmp") {
      DEBUG2 << "push_openmp() start";
      push_openmp();
      DEBUG2 << "push_openmp() end";
    } else if (option["thread"] == "taskflow") {
      DEBUG2 << "push_taskflow() start";
      push_taskflow();
      DEBUG2 << "push_taskflow() end";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    DEBUG2 << "push() end";
    float64 wclock2 = nix::wall_clock();

    json log = {{"elapsed", wclock2 - wclock1}};
    logger->append(curstep, "push", log);
  }

  virtual void calculate_moment()
  {
    if (curstep == momstep)
      return;

    auto option = cfgparser->get_application()["option"];

    if (option.contains("thread") == false || option["thread"] == "openmp") {
      calculate_moment_openmp();
    } else if (option["thread"] == "taskflow") {
      calculate_moment_taskflow();
    }

    // cache
    momstep = curstep;
  }
};

#define DEFINE_MEMBER(type, name)                                                                  \
  template <typename Chunk>                                                                        \
  type PicApplication<Chunk>::name

DEFINE_MEMBER(, PicApplication)
(int argc, char** argv) : base_type(argc, argv), Ns(1), momstep(-1)
{
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  base_type::initialize(argc, argv);

  // get number of species
  Ns = cfgparser->get_parameter()["Ns"];

  // diagnostics
  diagnoser = create_diagnoser();

  if (cfgparser->get_diagnostic().is_array() == false) {
    ERROR << tfm::format("Invalid diagnostic");
  }

  // initialize communicators
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          MPI_Comm_dup(MPI_COMM_WORLD, &mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }
}

DEFINE_MEMBER(void, set_chunk_communicator)()
{
  for (int i = 0; i < chunkvec.size(); i++) {
    for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            chunkvec[i]->set_mpi_communicator(mode, iz, iy, ix, mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, setup_chunks)()
{
  base_type::setup_chunks();
  set_chunk_communicator();

  // apply boundary condition just in case
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_pack(Chunk::BoundaryEmf);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
      chunkvec[i]->set_boundary_unpack(Chunk::BoundaryEmf);
    }
  }
}

DEFINE_MEMBER(json, to_json)()
{
  json state = base_type::to_json();

  state["Ns"]      = Ns;
  state["momstep"] = momstep;

  return state;
}

DEFINE_MEMBER(bool, from_json)(json& state)
{
  if (base_type::from_json(state) == false) {
    return false;
  }

  Ns      = state["Ns"];
  momstep = state["momstep"];

  return true;
}

DEFINE_MEMBER(bool, rebalance)()
{
  if (base_type::rebalance()) {
    set_chunk_communicator();
    return true;
  }

  return false;
}

DEFINE_MEMBER(void, finalize)()
{
  // free MPI communicator
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          MPI_Comm_free(&mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }

  // free diagnoser (internal MPI communicator)
  diagnoser.reset();

  // finalize
  base_type::finalize();
}

DEFINE_MEMBER(void, diagnostic)()
{
  DEBUG2 << "diagnostic() start";
  float64 wclock1 = nix::wall_clock();

  json config = cfgparser->get_diagnostic();
  auto data   = this->get_internal_data();

  for (json::iterator it = config.begin(); it != config.end(); ++it) {
    diagnoser->diagnose(*it, *this, data);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  DEBUG2 << "diagnostic() end";
  float64 wclock2 = nix::wall_clock();

  json log = {{"elapsed", wclock2 - wclock1}};
  logger->append(curstep, "diagnostic", log);
}

DEFINE_MEMBER(void, push_taskflow)()
{
  const float64 delt = cfgparser->get_delt();

  tf::Executor executor(nix::get_max_threads());
  tf::Taskflow taskflow;

  // critical section if MPI_THREAD_MULTIPLE is not supported
  tf::CriticalSection critical_section;

  auto push1 = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto task = subflow.emplace([&]() {
        chunk->reset_load();
        chunk->push_bfd(0.5 * delt);
        chunk->push_velocity(delt);
        chunk->push_position(delt);
        chunk->deposit_current(delt);

        // packing for boundary exchange
        chunk->set_boundary_pack(Chunk::BoundaryCur);
        chunk->set_boundary_pack(Chunk::BoundaryParticle);

        chunk->push_bfd(0.5 * delt);
      });

      // boundary exchange
      auto bc_begin = subflow.emplace([&]() {
        chunk->set_boundary_begin(Chunk::BoundaryCur);
        chunk->set_boundary_begin(Chunk::BoundaryParticle);
      });

      // dependency
      task.precede(bc_begin);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(bc_begin);
      }
    }
  });

  auto push2 = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto task = subflow.emplace([&]() {
        chunk->push_efd(delt);
        chunk->set_boundary_pack(Chunk::BoundaryEmf);
      });

      // boundary exchange
      auto bc_begin = subflow.emplace([&]() { chunk->set_boundary_begin(Chunk::BoundaryEmf); });

      // dependency
      task.precede(bc_begin);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(bc_begin);
      }
    }
  });

  auto particle_bc_probe = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto start = subflow.emplace([&]() {});
      auto probe = subflow.emplace(
          [&]() { return chunk->set_boundary_probe(Chunk::BoundaryParticle, false); });
      auto end = subflow.emplace([&]() {});

      // dependency
      start.precede(probe);
      probe.precede(probe, end);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(probe);
      }
    }
  });

  auto particle_bc_end = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto start = subflow.emplace([&]() {});
      auto query =
          subflow.emplace([&]() { return chunk->set_boundary_query(Chunk::BoundaryParticle, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(Chunk::BoundaryParticle); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(Chunk::BoundaryParticle); });

      // dependency
      start.precede(query);
      query.precede(query, end);
      end.precede(unpack);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(query);
        critical_section.add(end);
      }
    }
  });

  auto cur_bc_end = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto start = subflow.emplace([&]() {});
      auto query =
          subflow.emplace([&]() { return chunk->set_boundary_query(Chunk::BoundaryCur, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(Chunk::BoundaryCur); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(Chunk::BoundaryCur); });

      // dependency
      start.precede(query);
      query.precede(query, end);
      end.precede(unpack);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(query);
        critical_section.add(end);
      }
    }
  });

  auto emf_bc_end = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto start = subflow.emplace([&]() {});
      auto query =
          subflow.emplace([&]() { return chunk->set_boundary_query(Chunk::BoundaryEmf, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(Chunk::BoundaryEmf); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(Chunk::BoundaryEmf); });

      // dependency
      start.precede(query);
      query.precede(query, end);
      end.precede(unpack);

      if (NIX_MPI_THREAD_LEVEL != MPI_THREAD_MULTIPLE) {
        critical_section.add(query);
        critical_section.add(end);
      }
    }
  });

  // dependency
  push1.precede(cur_bc_end);
  push1.precede(particle_bc_probe);
  push2.succeed(cur_bc_end);
  push2.precede(emf_bc_end);
  particle_bc_probe.precede(particle_bc_end);

  executor.run(taskflow).wait();
}

DEFINE_MEMBER(void, calculate_moment_taskflow)()
{
  tf::Executor executor(nix::get_max_threads());
  tf::Taskflow taskflow;

  auto deposit = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto local = subflow.emplace([&]() { chunk->deposit_moment(); });
      auto begin = subflow.emplace([&]() {
        chunk->set_boundary_pack(Chunk::BoundaryMom);
        chunk->set_boundary_begin(Chunk::BoundaryMom);
      });

      local.precede(begin);
    }
  });

  auto bc_end = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto local = subflow.emplace([&]() {
        chunk->set_boundary_end(Chunk::BoundaryMom);
        chunk->set_boundary_unpack(Chunk::BoundaryMom);
      });
    }
  });

  deposit.precede(bc_end);

  executor.run(taskflow).wait();
}

DEFINE_MEMBER(void, push_openmp)()
{
#pragma omp parallel
  {
    const float64 delt = cfgparser->get_delt();

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      // reset load
      chunkvec[i]->reset_load();

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);

      // push particle
      chunkvec[i]->push_velocity(delt);
      chunkvec[i]->push_position(delt);

      // calculate current
      chunkvec[i]->deposit_current(delt);

      // begin boundary exchange for current
      chunkvec[i]->set_boundary_pack(Chunk::BoundaryCur);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryCur);

      // begin boundary exchange for particle
      chunkvec[i]->set_boundary_pack(Chunk::BoundaryParticle);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryParticle);

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryCur);
      chunkvec[i]->set_boundary_unpack(Chunk::BoundaryCur);

      // push E
      chunkvec[i]->push_efd(delt);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_pack(Chunk::BoundaryEmf);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_probe(Chunk::BoundaryParticle, true);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryParticle);
      chunkvec[i]->set_boundary_unpack(Chunk::BoundaryParticle);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
      chunkvec[i]->set_boundary_unpack(Chunk::BoundaryEmf);
    }
  }
}

DEFINE_MEMBER(void, calculate_moment_openmp)()
{
#pragma parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->deposit_moment();
      chunkvec[i]->set_boundary_pack(Chunk::BoundaryMom);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryMom);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryMom);
      chunkvec[i]->set_boundary_unpack(Chunk::BoundaryMom);
    }
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
