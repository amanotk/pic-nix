// -*- C++ -*-
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

#include "diag/field.hpp"
#include "diag/history.hpp"
#include "diag/load.hpp"
#include "diag/particle.hpp"
#include "diag/resource.hpp"
#include "diag/tracer.hpp"

#include <taskflow/taskflow.hpp>

PicApplication::PicApplication(int argc, char** argv) : base_type(argc, argv), Ns(1), momstep(-1)
{
}

int PicApplication::get_num_species() const
{
  return Ns;
}

void PicApplication::calculate_moment()
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

void PicApplication::initialize(int argc, char** argv)
{
  base_type::initialize(argc, argv);

  // get number of species
  Ns = cfgparser->get_parameter()["Ns"];

  // initialize communicators
  for (int mode = 0; mode < NumBoundaryMode; mode++) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          MPI_Comm_dup(MPI_COMM_WORLD, &mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }
}

void PicApplication::initialize_diagnostic()
{
  if (cfgparser->get_diagnostic().is_array() == false) {
    ERROR << tfm::format("Invalid diagnostic");
  }

  base_type::initialize_diagnostic();

  diagvec.push_back(std::make_unique<HistoryDiag>(*this));
  diagvec.push_back(std::make_unique<ResourceDiag>(*this));
  diagvec.push_back(std::make_unique<LoadDiag>(*this));
  diagvec.push_back(std::make_unique<FieldDiag>(*this));
  diagvec.push_back(std::make_unique<ParticleDiag>(*this));
  diagvec.push_back(std::make_unique<PickupTracerDiag>(*this));
  diagvec.push_back(std::make_unique<TracerDiag>(*this));
}

void PicApplication::set_chunk_communicator()
{
  for (int i = 0; i < chunkvec.size(); i++) {
    for (int mode = 0; mode < NumBoundaryMode; mode++) {
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

void PicApplication::setup_chunks()
{
  base_type::setup_chunks();
  set_chunk_communicator();

  // apply boundary condition just in case
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_pack(BoundaryEmf);
      chunkvec[i]->set_boundary_begin(BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(BoundaryEmf);
      chunkvec[i]->set_boundary_unpack(BoundaryEmf);
    }
  }
}

bool PicApplication::rebalance()
{
  if (base_type::rebalance()) {
    set_chunk_communicator();
    return true;
  }

  return false;
}

void PicApplication::finalize()
{
  // free MPI communicator
  for (int mode = 0; mode < NumBoundaryMode; mode++) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          MPI_Comm_free(&mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }

  // finalize
  base_type::finalize();
}

std::string PicApplication::get_basedir()
{
  auto tmpdir  = std::getenv("PICNIX_TMPDIR");
  auto basedir = cfgparser->get_application().value("basedir", "");

  if (tmpdir == nullptr) {
    return basedir;
  } else {
    return std::filesystem::path(tmpdir) / basedir;
  }
}

json PicApplication::to_json()
{
  json state = base_type::to_json();

  state["Ns"]      = Ns;
  state["momstep"] = momstep;

  return state;
}

bool PicApplication::from_json(json& state)
{
  if (base_type::from_json(state) == false) {
    return false;
  }

  Ns      = state["Ns"];
  momstep = state["momstep"];

  return true;
}

void PicApplication::push()
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

void PicApplication::push_openmp()
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
      chunkvec[i]->set_boundary_pack(BoundaryCur);
      chunkvec[i]->set_boundary_begin(BoundaryCur);

      // begin boundary exchange for particle
      chunkvec[i]->set_boundary_pack(BoundaryParticle);
      chunkvec[i]->set_boundary_begin(BoundaryParticle);

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(BoundaryCur);
      chunkvec[i]->set_boundary_unpack(BoundaryCur);

      // push E
      chunkvec[i]->push_efd(delt);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_pack(BoundaryEmf);
      chunkvec[i]->set_boundary_begin(BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_probe(BoundaryParticle, true);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(BoundaryParticle);
      chunkvec[i]->set_boundary_unpack(BoundaryParticle);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(BoundaryEmf);
      chunkvec[i]->set_boundary_unpack(BoundaryEmf);
    }
  }
}

void PicApplication::calculate_moment_openmp()
{
#pragma parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->deposit_moment();
      chunkvec[i]->set_boundary_pack(BoundaryMom);
      chunkvec[i]->set_boundary_begin(BoundaryMom);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(BoundaryMom);
      chunkvec[i]->set_boundary_unpack(BoundaryMom);
    }
  }
}

void PicApplication::push_taskflow()
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
        chunk->set_boundary_pack(BoundaryCur);
        chunk->set_boundary_pack(BoundaryParticle);

        chunk->push_bfd(0.5 * delt);
      });

      // boundary exchange
      auto bc_begin = subflow.emplace([&]() {
        chunk->set_boundary_begin(BoundaryCur);
        chunk->set_boundary_begin(BoundaryParticle);
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
        chunk->set_boundary_pack(BoundaryEmf);
      });

      // boundary exchange
      auto bc_begin = subflow.emplace([&]() { chunk->set_boundary_begin(BoundaryEmf); });

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
      auto probe =
          subflow.emplace([&]() { return chunk->set_boundary_probe(BoundaryParticle, false); });
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
          subflow.emplace([&]() { return chunk->set_boundary_query(BoundaryParticle, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(BoundaryParticle); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(BoundaryParticle); });

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
      auto start  = subflow.emplace([&]() {});
      auto query  = subflow.emplace([&]() { return chunk->set_boundary_query(BoundaryCur, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(BoundaryCur); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(BoundaryCur); });

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
      auto start  = subflow.emplace([&]() {});
      auto query  = subflow.emplace([&]() { return chunk->set_boundary_query(BoundaryEmf, 0); });
      auto end    = subflow.emplace([&]() { chunk->set_boundary_end(BoundaryEmf); });
      auto unpack = subflow.emplace([&]() { chunk->set_boundary_unpack(BoundaryEmf); });

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

void PicApplication::calculate_moment_taskflow()
{
  tf::Executor executor(nix::get_max_threads());
  tf::Taskflow taskflow;

  auto deposit = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto local = subflow.emplace([&]() { chunk->deposit_moment(); });
      auto begin = subflow.emplace([&]() {
        chunk->set_boundary_pack(BoundaryMom);
        chunk->set_boundary_begin(BoundaryMom);
      });

      local.precede(begin);
    }
  });

  auto bc_end = taskflow.emplace([&](tf::Subflow& subflow) {
    for (auto& chunk : chunkvec) {
      auto local = subflow.emplace([&]() {
        chunk->set_boundary_end(BoundaryMom);
        chunk->set_boundary_unpack(BoundaryMom);
      });
    }
  });

  deposit.precede(bc_end);

  executor.run(taskflow).wait();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
