// -*- C++ -*-
#ifndef _EXPIC3D_HPP_
#define _EXPIC3D_HPP_

#include "exchunk3d.hpp"
#include "nix/application.hpp"
#include "nix/chunkmap.hpp"
#include "nix/jsonio.hpp"

//
template <int Order>
class ExPIC3D : public Application<ExChunk3D<Order>, ChunkMap<3>>
{
protected:
  using json       = nlohmann::ordered_json;
  using BaseApp    = Application<ExChunk3D<Order>, ChunkMap<3>>;
  using Chunk      = ExChunk3D<Order>;
  using MpiCommVec = std::vector<MPI_Comm>;
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
  using BaseApp::delh;
  using BaseApp::cc;
  using BaseApp::xlim;
  using BaseApp::ylim;
  using BaseApp::zlim;
  using BaseApp::periodic;
  using BaseApp::nprocess;
  using BaseApp::thisrank;
  using BaseApp::bufsize;
  using BaseApp::sendbuf;
  using BaseApp::recvbuf;

  int        rebuild_interval; ///< interval for rebuild_chankmap
  int        Ns;               ///< number of species
  MpiCommVec mpicommvec;       ///< MPI Communicators

  virtual void parse_cfg() override;

  virtual void diagnostic_load(std::ostream &out, json &obj);

  virtual void diagnostic_field(std::ostream &out, json &obj);

  virtual void diagnostic_particle(std::ostream &out, json &obj);

  virtual void initialize(int argc, char **argv) override;

  virtual void setup() override;

  virtual void rebuild_chunkmap() override;

public:
  ExPIC3D(int argc, char **argv);

  virtual void push() override;

  virtual void diagnostic(std::ostream &out) override;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
