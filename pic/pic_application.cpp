// -*- C++ -*-
#include "pic_application.hpp"

#include "pic_chunk.hpp"
#include "pic_diag.hpp"
#include "pic_poisson.hpp"
#include "pic_poisson_factory.hpp"

#include "diag/field.hpp"
#include "diag/history.hpp"
#include "diag/particle.hpp"
#include "diag/tracer.hpp"
#include "diag/tracer_pickup.hpp"

#if PICNIX_ENABLE_ASCENT
#include "diag/ascent.hpp"
#endif

#include "nix/diag/load.hpp"
#include "nix/diag/resource.hpp"

PicApplication::PtrPoissonInterface PicApplication::create_poisson_interface()
{
  nix::Dims3D dims = {ndims[0], ndims[1], ndims[2]};
  float64     delh = cfgparser->get_delx();

  return std::static_pointer_cast<elliptic::SolverInterface>(make_poisson_solver(dims, delh));
}

int PicApplicationInterface::get_num_species()
{
  return static_cast<PicApplication*>(app_pointer)->get_num_species();
}

void PicApplicationInterface::calculate_moment()
{
  static_cast<PicApplication*>(app_pointer)->calculate_moment();
}

PicApplication::PicApplication(int argc, char** argv, PtrInterface interface)
    : base_type(argc, argv, interface), Ns(1), momstep(-1)
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

  calculate_moment_openmp();

  // cache
  momstep = curstep;
}

void PicApplication::initialize(int argc, char** argv)
{
  base_type::initialize(argc, argv);

  if (elliptic::Solver::initialize(&argc, &argv) != 0) {
    ERROR << "Failed to initialize elliptic solver backend." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // get number of species
  Ns = cfgparser->get_parameter()["Ns"];

  assert_mpi(performance.configure(cfgparser->get_application()),
             "invalid `application.performance` configuration");

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

  // initialize Poisson solver
  {
    auto poisson = create_poisson_interface();
    solver       = std::make_unique<elliptic::Solver>(poisson);

    if (solver != nullptr) {
      solver->set_option(cfgparser->get_application());
    }
  }
}

void PicApplication::initialize_diagnostic()
{
  const auto diagnostics = cfgparser->get_diagnostic();
  if (diagnostics.is_array() == false) {
    ERROR << fmt::format("Invalid diagnostic");
  }

#if PICNIX_ENABLE_ASCENT
  int ascent_count = 0;
  for (const auto& diagnostic : diagnostics) {
    if (diagnostic.value("name", std::string{}) == "ascent") {
      ascent_count++;
    }
  }
  if (ascent_count > 1) {
    ERROR << "Only one Ascent diagnostic entry is supported";
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#endif

  base_type::initialize_diagnostic();

  auto interface = std::static_pointer_cast<PicApplicationInterface>(get_interface());
  diagvec.push_back(std::make_unique<HistoryDiag>(interface));
  diagvec.push_back(std::make_unique<nix::ResourceDiag<PicDiag>>(interface));
  diagvec.push_back(std::make_unique<nix::LoadDiag<PicDiag, PicPacker>>(interface));
  diagvec.push_back(std::make_unique<FieldDiag>(interface));
  diagvec.push_back(std::make_unique<ParticleDiag>(interface));
  diagvec.push_back(std::make_unique<TracerPickupDiag>(interface));
  diagvec.push_back(std::make_unique<TracerDiag>(interface));
#if PICNIX_ENABLE_ASCENT
  diagvec.push_back(std::make_unique<AscentDiag>(interface));
#endif
}

void PicApplication::set_chunk_communicator()
{
  for (int i = 0; i < chunkvec.size(); i++) {
    auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

    for (int mode = 0; mode < NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            chunk->set_mpi_communicator(mode, iz, iy, ix, mpicommvec(mode, iz, iy, ix));
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

  if (get_mpi_thread_mode() == nix::MpiThreadMode::Multiple) {
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
      for (int i = 0; i < chunkvec.size(); i++) {
        auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

        chunk->set_boundary_pack(BoundaryEmf);
        chunk->set_boundary_begin(BoundaryEmf);
      }

#pragma omp for schedule(dynamic)
      for (int i = 0; i < chunkvec.size(); i++) {
        auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

        chunk->set_boundary_end(BoundaryEmf);
        chunk->set_boundary_unpack(BoundaryEmf);
      }
    }
    return;
  }

#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      static_cast<PicChunk*>(chunkvec[i].get())->set_boundary_pack(BoundaryEmf);
    }

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        static_cast<PicChunk*>(chunk_ptr.get())->set_boundary_begin(BoundaryEmf);
      }
    }
#pragma omp barrier

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        static_cast<PicChunk*>(chunk_ptr.get())->set_boundary_end(BoundaryEmf);
      }
    }
#pragma omp barrier

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      static_cast<PicChunk*>(chunkvec[i].get())->set_boundary_unpack(BoundaryEmf);
    }
  }
}

void PicApplication::setup_chunks_init()
{
  base_type::setup_chunks_init();

  // extract per-species q/m for serialization
  if (chunkvec.size() > 0) {
    auto data = static_cast<PicChunk*>(chunkvec[0].get())->get_internal_data();

    // check the validity of up
    bool is_up_valid = data.up.size() == Ns;
    if (is_up_valid) {
      for (int is = 0; is < Ns; is++) {
        is_up_valid = is_up_valid && (data.up[is] != nullptr);
      }
    }
    if (!is_up_valid) {
      ERROR << fmt::format("Per-species q/m extraction failed: up is null for some species");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // extract per-species q/m
    qm_per_species.resize(Ns);
    for (int is = 0; is < Ns; is++) {
      qm_per_species[is] = data.up[is]->q / data.up[is]->m;
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
  solver.reset();

  if (elliptic::Solver::finalize() != 0) {
    ERROR << "Failed to finalize elliptic solver backend." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

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
  state["qm"]      = qm_per_species;

  return state;
}

bool PicApplication::from_json(json& state)
{
  if (base_type::from_json(state) == false) {
    return false;
  }

  Ns      = state["Ns"];
  momstep = state["momstep"];

  if (state.contains("qm") == true) {
    qm_per_species = state["qm"].get<std::vector<float64>>();
  }

  return true;
}

void PicApplication::push()
{
  DEBUG2 << "push() start";
  float64 wclock1 = nix::wall_clock();

  bool sample_performance = performance.begin_step(curstep, nthread);

  DEBUG2 << "push_openmp() start";
  if (get_mpi_thread_mode() == nix::MpiThreadMode::Multiple) {
    push_openmp_multiple();
  } else {
    push_openmp_funneled();
  }
  DEBUG2 << "push_openmp() end";

  float64 push_end      = nix::wall_clock();
  float64 barrier_begin = push_end;

  MPI_Barrier(MPI_COMM_WORLD);
  float64 barrier_end = nix::wall_clock();

  DEBUG2 << "push() end";
  float64 wclock2 = nix::wall_clock();

  json log = {{"elapsed", wclock2 - wclock1}};
  logger->append(curstep, "push", log);

  if (sample_performance) {
    json performance_log =
        performance.finish_step(push_end - wclock1, barrier_end - barrier_begin, MPI_COMM_WORLD);
    logger->append(curstep, "performance", performance_log);
  }
}

void PicApplication::push_openmp_multiple()
{
#pragma omp parallel
  {
    const float64 delt = cfgparser->get_delt();

    if (performance.is_sampling()) {
      int parallel_threads = 1;
#ifdef _OPENMP
      parallel_threads = omp_get_num_threads();
#endif
#pragma omp single
      performance.set_parallel_threads(parallel_threads);
    }

    performance.begin_wall(PicPerformance::Phase::Advance);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::Advance);

      // reset load
      chunk->reset_load();

      // push B for a half step
      chunk->push_bfd(0.5 * delt);

      // push particle
      chunk->push_velocity(delt);
      chunk->push_position(delt);

      // calculate current
      chunk->deposit_current(delt);

      // begin boundary exchange for current
      chunk->set_boundary_pack(BoundaryCur);
      performance.begin_operation(PicPerformance::Operation::CurrentBegin);
      chunk->set_boundary_begin(BoundaryCur);
      performance.end_operation(PicPerformance::Operation::CurrentBegin);

      // begin boundary exchange for particle
      chunk->set_boundary_pack(BoundaryParticle);
      performance.begin_operation(PicPerformance::Operation::ParticleBegin);
      chunk->set_boundary_begin(BoundaryParticle);
      performance.end_operation(PicPerformance::Operation::ParticleBegin);

      // push B for a half step
      chunk->push_bfd(0.5 * delt);

      performance.end_chunk(PicPerformance::Phase::Advance);
    }

    performance.end_wall(PicPerformance::Phase::Advance);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::CurrentField);

      performance.begin_operation(PicPerformance::Operation::CurrentWaitall);
      chunk->set_boundary_end(BoundaryCur);
      performance.end_operation(PicPerformance::Operation::CurrentWaitall);
      chunk->set_boundary_unpack(BoundaryCur);

      // push E
      chunk->push_efd(delt);

      // begin boundary exchange for field
      chunk->set_boundary_pack(BoundaryEmf);
      performance.begin_operation(PicPerformance::Operation::FieldBegin);
      chunk->set_boundary_begin(BoundaryEmf);
      performance.end_operation(PicPerformance::Operation::FieldBegin);

      performance.end_chunk(PicPerformance::Phase::CurrentField);
    }

    performance.end_wall(PicPerformance::Phase::CurrentField);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::ParticleProbe);

      performance.begin_operation(PicPerformance::Operation::ParticleProbe);
      chunk->set_boundary_probe(BoundaryParticle, true);
      performance.end_operation(PicPerformance::Operation::ParticleProbe);

      performance.end_chunk(PicPerformance::Phase::ParticleProbe);
    }

    performance.end_wall(PicPerformance::Phase::ParticleProbe);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::ParticleExchange);

      performance.begin_operation(PicPerformance::Operation::ParticleWaitall);
      chunk->set_boundary_end(BoundaryParticle);
      performance.end_operation(PicPerformance::Operation::ParticleWaitall);
      chunk->set_boundary_unpack(BoundaryParticle);

      performance.end_chunk(PicPerformance::Phase::ParticleExchange);
    }

    performance.end_wall(PicPerformance::Phase::ParticleExchange);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::FieldExchange);

      performance.begin_operation(PicPerformance::Operation::FieldWaitall);
      chunk->set_boundary_end(BoundaryEmf);
      performance.end_operation(PicPerformance::Operation::FieldWaitall);
      chunk->set_boundary_unpack(BoundaryEmf);

      performance.end_chunk(PicPerformance::Phase::FieldExchange);
    }

    performance.end_wall(PicPerformance::Phase::FieldExchange);
  }
}

void PicApplication::complete_boundaries_funneled(int mode, PicPerformance::Operation operation)
{
  std::vector<char> complete(chunkvec.size(), false);
  size_t            incomplete = complete.size();

  while (incomplete > 0) {
    // Thread 0 must keep entering MPI because no other thread can drive progress
    // under MPI_THREAD_FUNNELED.
    for (size_t i = 0; i < complete.size(); i++) {
      if (complete[i]) {
        continue;
      }

      auto* chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_operation(operation);
      bool send_complete = chunk->set_boundary_query(mode, +1);
      performance.end_operation(operation);

      performance.begin_operation(operation);
      bool recv_complete = chunk->set_boundary_query(mode, -1);
      performance.end_operation(operation);

      if (send_complete && recv_complete) {
        performance.begin_operation(operation);
        chunk->set_boundary_end(mode);
        performance.end_operation(operation);
        complete[i] = true;
        incomplete--;
      }
    }
  }
}

void PicApplication::push_openmp_funneled()
{
  std::vector<char> particle_ready(chunkvec.size(), false);

#pragma omp parallel
  {
    const float64 delt = cfgparser->get_delt();

    if (performance.is_sampling()) {
#pragma omp master
      performance.set_parallel_threads(nix::get_num_threads());
#pragma omp barrier
    }

    performance.begin_wall(PicPerformance::Phase::Advance);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::Advance);
      chunk->reset_load();
      chunk->push_bfd(0.5 * delt);
      chunk->push_velocity(delt);
      chunk->push_position(delt);
      chunk->deposit_current(delt);
      chunk->set_boundary_pack(BoundaryCur);
      chunk->set_boundary_pack(BoundaryParticle);
      chunk->push_bfd(0.5 * delt);
      performance.end_chunk(PicPerformance::Phase::Advance);
    }

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        auto* chunk = static_cast<PicChunk*>(chunk_ptr.get());
        performance.begin_operation(PicPerformance::Operation::CurrentBegin);
        chunk->set_boundary_begin(BoundaryCur);
        performance.end_operation(PicPerformance::Operation::CurrentBegin);
      }
      for (auto& chunk_ptr : chunkvec) {
        auto* chunk = static_cast<PicChunk*>(chunk_ptr.get());
        performance.begin_operation(PicPerformance::Operation::ParticleBegin);
        chunk->set_boundary_begin(BoundaryParticle);
        performance.end_operation(PicPerformance::Operation::ParticleBegin);
      }
    }
#pragma omp barrier

    performance.end_wall(PicPerformance::Phase::Advance);

#pragma omp master
    {
      complete_boundaries_funneled(BoundaryCur, PicPerformance::Operation::CurrentPoll);
    }
#pragma omp barrier

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::CurrentField);
      chunk->set_boundary_unpack(BoundaryCur);
      chunk->push_efd(delt);
      chunk->set_boundary_pack(BoundaryEmf);
      performance.end_chunk(PicPerformance::Phase::CurrentField);
    }

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        auto* chunk = static_cast<PicChunk*>(chunk_ptr.get());
        performance.begin_operation(PicPerformance::Operation::FieldBegin);
        chunk->set_boundary_begin(BoundaryEmf);
        performance.end_operation(PicPerformance::Operation::FieldBegin);
      }
    }
#pragma omp barrier

    performance.end_wall(PicPerformance::Phase::CurrentField);

#pragma omp master
    {
      size_t incomplete = particle_ready.size();
      performance.begin_operation(PicPerformance::Operation::ParticleProbe);
      while (incomplete > 0) {
        for (size_t i = 0; i < particle_ready.size(); i++) {
          if (particle_ready[i]) {
            continue;
          }
          auto* chunk = static_cast<PicChunk*>(chunkvec[i].get());
          if (chunk->set_boundary_probe(BoundaryParticle, false)) {
            particle_ready[i] = true;
            incomplete--;
          }
        }
      }
      performance.end_operation(PicPerformance::Operation::ParticleProbe);
    }
#pragma omp barrier

    performance.end_wall(PicPerformance::Phase::ParticleProbe);

#pragma omp master
    {
      complete_boundaries_funneled(BoundaryParticle, PicPerformance::Operation::ParticlePoll);
    }
#pragma omp barrier

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::ParticleExchange);
      chunk->set_boundary_unpack(BoundaryParticle);
      performance.end_chunk(PicPerformance::Phase::ParticleExchange);
    }

    performance.end_wall(PicPerformance::Phase::ParticleExchange);

#pragma omp master
    {
      complete_boundaries_funneled(BoundaryEmf, PicPerformance::Operation::FieldPoll);
    }
#pragma omp barrier

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

      performance.begin_chunk(PicPerformance::Phase::FieldExchange);
      chunk->set_boundary_unpack(BoundaryEmf);
      performance.end_chunk(PicPerformance::Phase::FieldExchange);
    }

    performance.end_wall(PicPerformance::Phase::FieldExchange);
  }
}

void PicApplication::calculate_moment_openmp()
{
  if (get_mpi_thread_mode() == nix::MpiThreadMode::Multiple) {
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
      for (int i = 0; i < chunkvec.size(); i++) {
        auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

        chunk->deposit_moment();
        chunk->set_boundary_pack(BoundaryMom);
        chunk->set_boundary_begin(BoundaryMom);
      }

#pragma omp for schedule(dynamic)
      for (int i = 0; i < chunkvec.size(); i++) {
        auto chunk = static_cast<PicChunk*>(chunkvec[i].get());

        chunk->set_boundary_end(BoundaryMom);
        chunk->set_boundary_unpack(BoundaryMom);
      }
    }
    return;
  }

#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(chunkvec[i].get());
      chunk->deposit_moment();
      chunk->set_boundary_pack(BoundaryMom);
    }

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        static_cast<PicChunk*>(chunk_ptr.get())->set_boundary_begin(BoundaryMom);
      }
    }
#pragma omp barrier

#pragma omp master
    {
      for (auto& chunk_ptr : chunkvec) {
        static_cast<PicChunk*>(chunk_ptr.get())->set_boundary_end(BoundaryMom);
      }
    }
#pragma omp barrier

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      static_cast<PicChunk*>(chunkvec[i].get())->set_boundary_unpack(BoundaryMom);
    }
  }
}

void PicApplication::update_poisson_efield()
{
  if (solver == nullptr) {
    return;
  }

  auto poisson = std::dynamic_pointer_cast<PicPoisson>(solver->get_interface());
  if (poisson == nullptr) {
    ERROR << "PicApplication requires PicPoisson solver interface." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // bind chunks to poisson solver and get its accessor
  poisson->bind_chunks(chunkvec);
  auto accessor = poisson->get_accessor();

  // solve Poisson equation
  const int status = solver->solve(accessor);
  if (status != 0) {
    ERROR << "Poisson solve failed with status: " << status << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // compute E-field from potential
  exchange_phi_boundaries();

  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    chunk->compute_efield_poisson();
  }

  exchange_emf_boundaries();
}

void PicApplication::exchange_phi_boundaries()
{
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    chunk->set_boundary_pack(BoundaryPhi);
    chunk->set_boundary_begin(BoundaryPhi);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    chunk->set_boundary_end(BoundaryPhi);
    chunk->set_boundary_unpack(BoundaryPhi);
  }
}

void PicApplication::exchange_emf_boundaries()
{
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    chunk->set_boundary_pack(BoundaryEmf);
    chunk->set_boundary_begin(BoundaryEmf);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    chunk->set_boundary_end(BoundaryEmf);
    chunk->set_boundary_unpack(BoundaryEmf);
  }
}
