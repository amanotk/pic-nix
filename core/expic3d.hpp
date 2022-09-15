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
private:
  using json       = nlohmann::ordered_json;
  using BaseApp    = Application<ExChunk3D<Order>, ChunkMap<3>>;
  using Chunk      = ExChunk3D<Order>;
  using MpiCommVec = std::vector<MPI_Comm>;

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

  int        Ns;         ///< number of species
  MpiCommVec mpicommvec; ///< MPI Communicators

  virtual void parse_cfg() override;

  virtual void write_field_chunk(MPI_File &fh, json &dataset, size_t &disp, const char *name,
                                 const char *desc, const int size, const int ndim, const int *dims,
                                 const int mode);

  virtual void diagnostic_load(std::ostream &out, json &obj);

  virtual void diagnostic_field(std::ostream &out, json &obj);

  virtual void diagnostic_particle(std::ostream &out, json &obj);

  virtual void wait_bc_exchange(std::set<int> &queue, const int mode);

  virtual void initialize(int argc, char **argv) override;

  virtual void setup() override;

  virtual void rebuild_chunkmap() override;

public:
  ExPIC3D(int argc, char **argv) : BaseApp(argc, argv), Ns(1)
  {
  }

  virtual void push() override;

  virtual void diagnostic(std::ostream &out) override;

  virtual void finalize(int cleanup = 0) override
  {
    BaseApp::finalize(1);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
