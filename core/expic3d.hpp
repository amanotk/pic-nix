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

  int         Ns;                ///< number of species
  std::string datadir;           ///< data output directory
  std::string prefix_load;       ///< output filename prefix for load
  std::string prefix_history;    ///< output filename prefix for history
  std::string prefix_field;      ///< output filename prefix for field
  std::string prefix_particle;   ///< output filename prefix for particle
  int         interval_load;     ///< data output interval for load
  int         interval_history;  ///< data output interval for history
  int         interval_field;    ///< data output interval for field
  int         interval_particle; ///< data output interval for particle
  MpiCommVec  mpicommvec;        ///< MPI Communicators

  virtual void parse_cfg() override;

  virtual void write_field_chunk(MPI_File &fh, json &dataset, size_t &disp, const char *name,
                                 const char *desc, const int size, const int ndim, const int *dims,
                                 const int mode);

  virtual void diagnostic_load(std::ostream &out);

  virtual void diagnostic_history(std::ostream &out);

  virtual void diagnostic_field(std::ostream &out);

  virtual void diagnostic_particle(std::ostream &out);

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
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
