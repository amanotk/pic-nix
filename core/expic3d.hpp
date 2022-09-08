// -*- C++ -*-
#ifndef _EXPIC3D_HPP_
#define _EXPIC3D_HPP_

#include "exchunk3d.hpp"
#include "nix/application.hpp"
#include "nix/chunkmap.hpp"
#include "nix/jsonio.hpp"

//
template <int Nb>
class ExPIC3D : public Application<ExChunk3D<Nb>, ChunkMap<3>>
{
private:
  using BaseApp = Application<ExChunk3D<Nb>, ChunkMap<3>>;
  using Chunk   = ExChunk3D<Nb>;

protected:
  using typename BaseApp::json;
  using BaseApp::cfg_file;
  using BaseApp::cfg_json;
  using BaseApp::numchunk;
  using BaseApp::chunkvec;
  using BaseApp::chunkmap;
  using BaseApp::ndims;
  using BaseApp::cdims;
  using BaseApp::curstep;
  using BaseApp::curtime;
  using BaseApp::delt;
  using BaseApp::delh;
  using BaseApp::cc;
  using BaseApp::periodic;
  using BaseApp::nprocess;
  using BaseApp::thisrank;
  using BaseApp::bufsize;
  using BaseApp::sendbuf;
  using BaseApp::recvbuf;

  std::string prefix;   ///< output filename prefix
  int         interval; ///< data output interval

public:
  ExPIC3D(int argc, char **argv) : BaseApp(argc, argv)
  {
  }

  virtual void initialize(int argc, char **argv) override;

  virtual void push() override;

  virtual void diagnostic(std::ostream &out) override;

  virtual void wait_bc_exchange(std::set<int> &queue, const int mode);
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
