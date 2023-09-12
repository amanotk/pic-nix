// -*- C++ -*-
#ifndef _EXPIC3D_HPP_
#define _EXPIC3D_HPP_

#include "exchunk3d.hpp"
#include "nix/application.hpp"
#include "nix/chunkmap.hpp"

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
template <typename Chunk, typename Diagnoser>
class ExPIC3D : public nix::Application<Chunk, nix::ChunkMap<3>>
{
public:
  using ThisType     = ExPIC3D<Chunk, Diagnoser>;
  using BaseApp      = nix::Application<Chunk, nix::ChunkMap<3>>;
  using ChunkType    = Chunk;
  using PtrDiagnoser = std::unique_ptr<Diagnoser>;
  using MpiCommVec   = xt::xtensor_fixed<MPI_Comm, xt::xshape<Chunk::NumBoundaryMode, 3, 3, 3>>;

protected:
  using BaseApp::cfgparser;
  using BaseApp::argparser;
  using BaseApp::balancer;
  using BaseApp::logger;
  using BaseApp::chunkvec;
  using BaseApp::chunkmap;
  using BaseApp::ndims;
  using BaseApp::cdims;
  using BaseApp::curstep;
  using BaseApp::curtime;
  using BaseApp::delt;
  using BaseApp::delx;
  using BaseApp::dely;
  using BaseApp::delz;
  using BaseApp::xlim;
  using BaseApp::ylim;
  using BaseApp::zlim;
  using BaseApp::nprocess;
  using BaseApp::thisrank;

  int          Ns;         ///< number of species
  int          momstep;    ///< step at which moment quantities are cached
  MpiCommVec   mpicommvec; ///< MPI Communicators
  PtrDiagnoser diagnoser;  ///< diagnostic handler

  virtual void initialize(int argc, char** argv) override;

  virtual void set_chunk_communicator();

  virtual void setup_chunks() override;

  virtual bool rebalance() override;

  virtual void finalize() override;

  virtual std::unique_ptr<Chunk> create_chunk(const int dims[], int id) override = 0;

public:
  ExPIC3D(int argc, char** argv);

  virtual json to_json() override;

  virtual bool from_json(json& state) override;

  virtual void push() override;

  virtual void diagnostic() override;

  virtual void calculate_moment();

  int get_Ns()
  {
    return Ns;
  }
};

#define DEFINE_MEMBER(type, name)                                                                  \
  template <typename Chunk, typename Diagnoser>                                                    \
  type ExPIC3D<Chunk, Diagnoser>::name

DEFINE_MEMBER(, ExPIC3D)
(int argc, char** argv) : BaseApp(argc, argv), Ns(1), momstep(-1)
{
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  BaseApp::initialize(argc, argv);

  // get number of species
  Ns = cfgparser->get_parameter()["Ns"];

  // diagnostics
  diagnoser = std::make_unique<Diagnoser>(this->get_basedir());

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
  BaseApp::setup_chunks();
  set_chunk_communicator();

  // apply boundary condition just in case
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }
}

DEFINE_MEMBER(json, to_json)()
{
  json state = BaseApp::to_json();

  state["Ns"]      = Ns;
  state["momstep"] = momstep;

  return state;
}

DEFINE_MEMBER(bool, from_json)(json& state)
{
  if (BaseApp::from_json(state) == false) {
    return false;
  }

  Ns      = state["Ns"];
  momstep = state["momstep"];

  return true;
}

DEFINE_MEMBER(bool, rebalance)()
{
  if (BaseApp::rebalance()) {
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

  // finalize
  BaseApp::finalize();
}

DEFINE_MEMBER(void, push)()
{
  DEBUG2 << "push() start";
  float64 wclock1 = nix::wall_clock();

#pragma omp parallel
  {
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
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryCur);

      // begin boundary exchange for particle
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryParticle);

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryCur);

      // push E
      chunkvec[i]->push_efd(delt);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryParticle);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }

  DEBUG2 << "push() end";
  float64 wclock2 = nix::wall_clock();

  json log = {{"elapsed", wclock2 - wclock1}};
  logger->append(curstep, "push", log);
}

DEFINE_MEMBER(void, diagnostic)()
{
  DEBUG2 << "diagnostic() start";
  float64 wclock1 = nix::wall_clock();

  {
    json config = cfgparser->get_diagnostic();
    auto data   = this->get_internal_data();

    for (json::iterator it = config.begin(); it != config.end(); ++it) {
      diagnoser->doit(*it, *this, data);
    }
  }

  DEBUG2 << "diagnostic() end";
  float64 wclock2 = nix::wall_clock();

  json log = {{"elapsed", wclock2 - wclock1}};
  logger->append(curstep, "diagnostic", log);
}

DEFINE_MEMBER(void, calculate_moment)()
{
  if (curstep == momstep)
    return;

#pragma parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->deposit_moment();
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryMom);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < chunkvec.size(); i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryMom);
    }
  }

  // cache
  momstep = curstep;
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
