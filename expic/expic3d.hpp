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
template <int Order>
class ExPIC3D : public nix::Application<ExChunk3D<Order>, nix::ChunkMap<3>>
{
protected:
  using BaseApp    = nix::Application<ExChunk3D<Order>, nix::ChunkMap<3>>;
  using Chunk      = ExChunk3D<Order>;
  using MpiCommVec = xt::xtensor_fixed<MPI_Comm, xt::xshape<Chunk::NumBoundaryMode, 3, 3, 3>>;
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

  int        Ns;         ///< number of species
  int        momstep;    ///< step at which moment quantities are cached
  MpiCommVec mpicommvec; ///< MPI Communicators

  virtual void parse_cfg() override;

  virtual void diagnostic_field(std::ostream& out, json& obj);

  virtual void diagnostic_particle(std::ostream& out, json& obj);

  virtual void diagnostic_history(std::ostream& out, json& obj);

  virtual void calculate_moment();

  virtual void initialize(int argc, char** argv) override;

  virtual void set_chunk_communicator();

  virtual void setup() override;

  virtual bool rebuild_chunkmap() override;

  virtual std::unique_ptr<ExChunk3D<Order>> create_chunk(const int dims[], int id) override = 0;

public:
  ExPIC3D(int argc, char** argv);

  virtual void push() override;

  virtual void diagnostic(std::ostream& out) override;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
