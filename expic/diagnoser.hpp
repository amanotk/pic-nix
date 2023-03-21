// -*- C++ -*-
#ifndef _DIAGNOSER_HPP_
#define _DIAGNOSER_HPP_

#include "nix/application.hpp"
#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/particle.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Buffer;
using nix::Particle;

template <typename Data>
void write_chunk_all(Data& data, MPI_File& fh, size_t& disp, int mode)
{
  int bufsize = 0;

  for (int i = 0; i < data.numchunk; i++) {
    bufsize += data.chunkvec[i]->pack_diagnostic(mode, nullptr, 0);
  }

  // write to disk
  Buffer   buffer(bufsize);
  uint8_t* bufptr = buffer.get();

  // pack
  for (int i = 0, address = 0; i < data.numchunk; i++) {
    address = data.chunkvec[i]->pack_diagnostic(mode, bufptr, address);
  }

  // collective write
  jsonio::write_contiguous(&fh, &disp, bufptr, bufsize, 1, 1);
}

class HistoryDiagnoser
{
public:
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (data.curstep % config.value("interval", 1) != 0)
      return;

    const int            Ns = app.get_Ns();
    std::vector<float64> history(Ns + 4);

    // clear
    std::fill(history.begin(), history.end(), 0.0);

    // calculate moment if not cached
    app.calculate_moment();

    // calculate divergence error and energy
    for (int i = 0; i < data.numchunk; i++) {
      float64 div_e = 0;
      float64 div_b = 0;
      float64 ene_e = 0;
      float64 ene_b = 0;
      float64 ene_p[Ns];

      data.chunkvec[i]->get_diverror(div_e, div_b);
      data.chunkvec[i]->get_energy(ene_e, ene_b, ene_p);

      history[0] += div_e;
      history[1] += div_b;
      history[2] += ene_e;
      history[3] += ene_b;
      for (int is = 0; is < Ns; is++) {
        history[is + 4] += ene_p[is];
      }
    }

    {
      void* sndptr = history.data();
      void* rcvptr = nullptr;

      if (data.thisrank == 0) {
        sndptr = MPI_IN_PLACE;
        rcvptr = history.data();
      }

      MPI_Reduce(sndptr, rcvptr, Ns + 4, MPI_FLOAT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // output from root
    if (data.thisrank == 0) {
      // get parameters from json
      std::string prefix = config.value("prefix", "history");
      std::string path   = config.value("path", ".") + "/";
      std::string msg    = "";

      // initial call
      if (data.curstep == 0) {
        // header
        msg += tfm::format("# %8s %15s", "step", "time");
        msg += tfm::format(" %15s", "div(E)");
        msg += tfm::format(" %15s", "div(B)");
        msg += tfm::format(" %15s", "E^2/2");
        msg += tfm::format(" %15s", "B^2/2");
        for (int is = 0; is < Ns; is++) {
          msg += tfm::format("    Particle #%02d", is);
        }
        msg += "\n";

        // clear file
        std::ofstream ofs(path + prefix + ".txt", nix::text_write);
        ofs.close();
      }

      msg += tfm::format("  %8s %15.6e", nix::format_step(data.curstep), data.curtime);
      msg += tfm::format(" %15.6e", history[0]);
      msg += tfm::format(" %15.6e", history[1]);
      msg += tfm::format(" %15.6e", history[2]);
      msg += tfm::format(" %15.6e", history[3]);
      for (int is = 0; is < Ns; is++) {
        msg += tfm::format(" %15.6e", history[is + 4]);
      }
      msg += "\n";

      // output to steam
      std::cout << msg << std::flush;

      // append to file
      {
        std::ofstream ofs(path + prefix + ".txt", nix::text_append);
        ofs << msg << std::flush;
        ofs.close();
      }
    }
  }
};

class FieldDiagnoser
{
public:
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (data.curstep % config.value("interval", 1) != 0)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    // get parameters from json
    std::string prefix = config.value("prefix", "field");
    std::string path   = config.value("path", ".") + "/";

    // filename
    std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(data.curstep));
    std::string fn_json   = fn_prefix + ".json";
    std::string fn_data   = fn_prefix + ".data";

    MPI_File fh;
    size_t   disp;
    json     dataset;

    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");

    //
    // coordinate
    //
    {
      const char name[]  = "xc";
      const char desc[]  = "x coordinate";
      const int  ndim    = 2;
      const int  dims[2] = {nc, nx};
      const int  size    = nc * nx * sizeof(float64);

      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticX);
    }

    {
      const char name[]  = "yc";
      const char desc[]  = "y coordinate";
      const int  ndim    = 2;
      const int  dims[2] = {nc, ny};
      const int  size    = nc * ny * sizeof(float64);

      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticY);
    }

    {
      const char name[]  = "zc";
      const char desc[]  = "z coordinate";
      const int  ndim    = 2;
      const int  dims[2] = {nc, nz};
      const int  size    = nc * nz * sizeof(float64);

      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticZ);
    }

    //
    // electromagnetic field
    //
    {
      const char name[]  = "uf";
      const char desc[]  = "electromagnetic field";
      const int  ndim    = 5;
      const int  dims[5] = {nc, nz, ny, nx, 6};
      const int  size    = nc * nz * ny * nx * 6 * sizeof(float64);

      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticEmf);
    }

    //
    // current
    //
    {
      const char name[]  = "uj";
      const char desc[]  = "current";
      const int  ndim    = 5;
      const int  dims[5] = {nc, nz, ny, nx, 4};
      const int  size    = nc * nz * ny * nx * 4 * sizeof(float64);

      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticCur);
    }

    //
    // moment
    //
    {
      const char name[]  = "um";
      const char desc[]  = "moment";
      const int  ndim    = 6;
      const int  dims[6] = {nc, nz, ny, nx, Ns, 11};
      const int  size    = nc * nz * ny * nx * Ns * 11 * sizeof(float64);

      // calculate moment if not cached
      app.calculate_moment();

      // write
      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      write_chunk_all(data, fh, disp, App::Chunk::DiagnosticMom);
    }

    jsonio::close_file(&fh);

    //
    // output json file
    //
    {
      json root;

      // meta data
      root["meta"] = {{"endian", nix::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"order", 1},
                      {"time", data.curtime},
                      {"step", data.curstep}};
      // dataset
      root["dataset"] = dataset;

      if (data.thisrank == 0) {
        std::ofstream ofs(path + fn_json);
        ofs << std::setw(2) << root;
        ofs.close();
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
};

class ParticleDiagnoser
{
public:
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (data.curstep % config.value("interval", 1) != 0)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    // get parameters from json
    std::string prefix = config.value("prefix", "particle");
    std::string path   = config.value("path", ".") + "/";

    // filename
    std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(data.curstep));
    std::string fn_json   = fn_prefix + ".json";
    std::string fn_data   = fn_prefix + ".data";

    MPI_File fh;
    size_t   disp;
    json     dataset;

    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");

    //
    // for each particle
    //
    for (int is = 0; is < Ns; is++) {
      // write particles
      size_t disp0 = disp;
      int    mode  = App::Chunk::DiagnosticParticle + is;
      write_chunk_all(data, fh, disp, mode);

      // meta data
      {
        std::string name = tfm::format("up%02d", is);
        std::string desc = tfm::format("particle species %02d", is);

        const int Np      = (disp - disp0) / (Particle::Nc * sizeof(float64));
        const int ndim    = 2;
        const int dims[2] = {Np, Particle::Nc};
        const int size    = Np * Particle::Nc * sizeof(float64);

        jsonio::put_metadata(dataset, name.c_str(), "f8", desc.c_str(), disp0, size, ndim, dims);
      }
    }

    jsonio::close_file(&fh);

    //
    // output json file
    //
    {
      json root;

      // meta data
      root["meta"] = {{"endian", nix::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"order", 1},
                      {"time", data.curtime},
                      {"step", data.curstep}};
      // dataset
      root["dataset"] = dataset;

      if (data.thisrank == 0) {
        std::ofstream ofs(path + fn_json);
        ofs << std::setw(2) << root;
        ofs.close();
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
};

class Diagnoser
{
protected:
  HistoryDiagnoser  history;
  FieldDiagnoser    field;
  ParticleDiagnoser particle;

public:
  template <typename App, typename Data>
  void doit(json& config, App& app, Data& data)
  {
    if (config.contains("name") == false)
      return;

    if (config["name"] == "history") {
      history(config, app, data);
      return;
    }

    if (config["name"] == "field") {
      field(config, app, data);
      return;
    }

    if (config["name"] == "particle") {
      particle(config, app, data);
      return;
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
