// -*- C++ -*-
#ifndef _EXPIC3D_HPP_
#define _EXPIC3D_HPP_

#include "exchunk3d.hpp"
#include "nix/application.hpp"
#include "nix/chunkmap.hpp"
#include "nix/jsonio.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Balancer;
using nix::Particle;
using nix::ParticleVec;
using nix::PtrParticle;

///
/// @brief Application for 3D Explicit PIC Simulations
/// @tparam Order order of shape function
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
template <int Order, typename Diagnoser>
class ExPIC3D : public nix::Application<ExChunk3D<Order>, nix::ChunkMap<3>>
{
public:
  using ThisType     = ExPIC3D<Order, Diagnoser>;
  using BaseApp      = nix::Application<ExChunk3D<Order>, nix::ChunkMap<3>>;
  using Chunk        = ExChunk3D<Order>;
  using PtrDiagnoser = std::unique_ptr<Diagnoser>;
  using MpiCommVec   = xt::xtensor_fixed<MPI_Comm, xt::xshape<Chunk::NumBoundaryMode, 3, 3, 3>>;

protected:
  using BaseApp::cfg_file;
  using BaseApp::cfg_json;
  using BaseApp::balancer;
  using BaseApp::numchunk;
  using BaseApp::chunkvec;
  using BaseApp::chunkmap;
  using BaseApp::ndims;
  using BaseApp::cdims;
  using BaseApp::curstep;
  using BaseApp::curtime;
  using BaseApp::tmax;
  using BaseApp::delt;
  using BaseApp::delx;
  using BaseApp::dely;
  using BaseApp::delz;
  using BaseApp::cc;
  using BaseApp::xlim;
  using BaseApp::ylim;
  using BaseApp::zlim;
  using BaseApp::periodic;
  using BaseApp::nprocess;
  using BaseApp::thisrank;

  int          Ns;         ///< number of species
  int          momstep;    ///< step at which moment quantities are cached
  MpiCommVec   mpicommvec; ///< MPI Communicators
  PtrDiagnoser diagnoser;  ///< diagnostic handler

  virtual void parse_cfg() override;

  virtual void initialize(int argc, char** argv) override;

  virtual void set_chunk_communicator();

  virtual void setup() override;

  virtual bool rebuild_chunkmap() override;

  virtual std::unique_ptr<ExChunk3D<Order>> create_chunk(const int dims[], int id) override = 0;

public:
  ExPIC3D(int argc, char** argv);

  virtual void push() override;

  virtual void calculate_moment();

  virtual void diagnostic(std::ostream& out) override
  {
    json config = cfg_json["diagnostic"];
    auto data   = this->get_internal_data();

    for (json::iterator it = config.begin(); it != config.end(); ++it) {
      diagnoser->doit(*it, *this, data);
    }
  }

  int get_Ns()
  {
    return Ns;
  }
};

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order, typename Diagnoser>                                                         \
  type ExPIC3D<Order, Diagnoser>::name

DEFINE_MEMBER(, ExPIC3D)
(int argc, char** argv) : BaseApp(argc, argv), Ns(1), momstep(-1)
{
}

DEFINE_MEMBER(void, parse_cfg)()
{
  // read configuration file
  {
    std::ifstream f(cfg_file.c_str());
    cfg_json = json::parse(f, nullptr, true, true);
  }

  // application
  {
    json application = cfg_json["application"];
  }

  // parameters
  {
    json parameter = cfg_json["parameter"];

    // number of grids and chunks
    int nx = parameter["Nx"].get<int>();
    int ny = parameter["Ny"].get<int>();
    int nz = parameter["Nz"].get<int>();
    int cx = parameter["Cx"].get<int>();
    int cy = parameter["Cy"].get<int>();
    int cz = parameter["Cz"].get<int>();

    // other parameters
    float64 delh = parameter["delh"].get<float64>();

    Ns   = parameter["Ns"].get<int>();
    cc   = parameter["cc"].get<float64>();
    delt = parameter["delt"].get<float64>();
    delx = delh;
    dely = delh;
    delz = delh;

    // check dimensions
    if (!(nz % cz == 0 && ny % cy == 0 && nx % cx == 0)) {
      ERROR << tfm::format("Number of grid must be divisible by number of chunk");
      ERROR << tfm::format("Nx, Ny, Nz = [%4d, %4d, %4d]", nx, ny, nz);
      ERROR << tfm::format("Cx, Cy, Cz = [%4d, %4d, %4d]", cx, cy, cz);
      this->finalize(-1);
      exit(-1);
    }

    // global number of grid
    ndims[0] = nz;
    ndims[1] = ny;
    ndims[2] = nx;
    ndims[3] = ndims[0] * ndims[1] * ndims[2];

    // global number of chunk
    cdims[0] = cz;
    cdims[1] = cy;
    cdims[2] = cx;
    cdims[3] = cdims[0] * cdims[1] * cdims[2];

    // global domain size
    xlim[0] = 0;
    xlim[1] = delx * ndims[2];
    xlim[2] = xlim[1] - xlim[0];
    ylim[0] = 0;
    ylim[1] = dely * ndims[1];
    ylim[2] = ylim[1] - ylim[0];
    zlim[0] = 0;
    zlim[1] = delz * ndims[0];
    zlim[2] = zlim[1] - zlim[0];
  }

  // check diagnostic
  if (cfg_json["diagnostic"].is_array() == false) {
    ERROR << tfm::format("Invalid diagnostic");
  }
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  // parse command line arguments
  this->parse_cmd(argc, argv);

  // parse configuration file
  this->parse_cfg();

  // some initial setup
  curstep = 0;
  curtime = 0.0;
  this->initialize_mpi(&argc, &argv);
  this->initialize_logger();
  this->initialize_debugprinting(this->debug);
  this->initialize_chunkmap();

  // diagnoser
  diagnoser = std::make_unique<Diagnoser>();

  // load balancer
  balancer = this->create_balancer();

  // set auxiliary information for chunk
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->set_global_context(offset, ndims);
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
  for (int i = 0; i < numchunk; i++) {
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

DEFINE_MEMBER(void, setup)()
{
  // set MPI communicator for each mode
  set_chunk_communicator();

  // setup for each chunk with boundary condition
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      // setup for physical conditions
      chunkvec[i]->setup(cfg_json["parameter"]);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }
}
DEFINE_MEMBER(bool, rebuild_chunkmap)()
{
  if (BaseApp::rebuild_chunkmap()) {
    // reset MPI communicator for each mode
    set_chunk_communicator();
    return true;
  }

  return false;
}

DEFINE_MEMBER(void, push)()
{
  float64 wclock1, wclock2;

  wclock1 = nix::wall_clock();

  DEBUG2 << "push() is called";
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
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
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryCur);

      // push E
      chunkvec[i]->push_efd(delt);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryParticle);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }

  wclock2 = nix::wall_clock();

  // log
  DEBUG2 << "log in push";
  {
    json push = {{"start", wclock1}, {"end", wclock2}, {"elapsed", wclock2 - wclock1}};
    this->logger->append(curstep, "push", push);
  }
}

DEFINE_MEMBER(void, calculate_moment)()
{
  if (curstep == momstep)
    return;

#pragma parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->deposit_moment();
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryMom);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
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
