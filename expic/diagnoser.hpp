// -*- C++ -*-
#ifndef _DIAGNOSER_HPP_
#define _DIAGNOSER_HPP_

#include "nix/application.hpp"
#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/nixio.hpp"

#include "nix/xtensor_packer3d.hpp"
#include "nix/xtensor_particle.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Buffer;
using nix::XtensorPacker3D;
using nix::ParticlePtr;
using nix::ParticleVec;
using ParticleType = nix::ParticlePtr::element_type;

class AsynchronousDiagnoser
{
protected:
  MPI_File                 filehandle;
  bool                     is_opened;
  std::vector<Buffer>      buffer;
  std::vector<MPI_Request> request;

  // check if the output is required
  bool require_output(int curstep, json& config)
  {
    bool status    = curstep % config.value("interval", 1) == 0;
    bool completed = is_completed();

    if (status == true) {
      // make sure all the requests are completed
      wait_all();
    }

    if (status == false && completed == false) {
      // check if all the request are completed
      int flag = 0;
      MPI_Testall(request.size(), request.data(), &flag, MPI_STATUSES_IGNORE);
    }

    return status;
  }

public:
  // constructor
  AsynchronousDiagnoser(int size) : is_opened(false)
  {
    buffer.resize(size);
    request.resize(size);

    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
  }

  // open file
  void open_file(const char* filename, size_t* disp, const char* mode)
  {
    if (is_opened == false) {
      nixio::open_file(filename, &filehandle, disp, mode);
      is_opened = true;
    }
  }

  // close file
  void close_file()
  {
    assert(is_completed() == true);

    if (is_opened == true) {
      nixio::close_file(&filehandle);
      is_opened = false;
    }
  }

  // check if all the requests are completed
  bool is_completed()
  {
    bool status = std::all_of(request.begin(), request.end(),
                              [](auto& req) { return req == MPI_REQUEST_NULL; });
    return status;
  }

  // wait for the completion of the job
  void wait(int jobid)
  {
    MPI_Wait(&request[jobid], MPI_STATUS_IGNORE);
  }

  // wait for the completion of all the jobs and close the file
  void wait_all()
  {
    MPI_Waitall(request.size(), request.data(), MPI_STATUSES_IGNORE);
    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
    close_file();
  }

  // launch asynchronous write
  template <typename DataPacker, typename Data>
  void launch(int jobid, DataPacker packer, Data& data, size_t& disp)
  {
    int bufsize = 0;

    // calculate buffer size
    for (int i = 0; i < data.chunkvec.size(); i++) {
      bufsize += data.chunkvec[i]->pack_diagnostic(packer, nullptr, 0);
    }

    // pack data
    buffer[jobid].resize(bufsize);
    uint8_t* bufptr = buffer[jobid].get();

    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      address = data.chunkvec[i]->pack_diagnostic(packer, bufptr, address);
    }

    // write to the disk
    nixio::write_contiguous(&filehandle, &disp, bufptr, bufsize, 1, 1, &request[jobid]);
    wait(jobid);
  }
};

///
/// @brief Diagnoser for time history
///
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
    for (int i = 0; i < data.chunkvec.size(); i++) {
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

///
/// @brief Diagnoser for computational work load
///
class LoadDiagnoser : public AsynchronousDiagnoser
{
protected:
  // data packer for load
  class LoadPacker
  {
  public:
    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      auto& load = data.load;

      int count = sizeof(float64) * load.size() + address;

      if (buffer == nullptr) {
        return count;
      }

      // packing
      float64* ptr = reinterpret_cast<float64*>(buffer + address);
      std::copy(load.begin(), load.end(), ptr);

      return count;
    }
  };

  // data packer for rank
  class RankPacker
  {
  private:
    int thisrank;

  public:
    RankPacker(int rank) : thisrank(rank)
    {
    }

    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      int count = sizeof(int) + address;

      if (buffer == nullptr) {
        return count;
      }

      // packing
      int* ptr = reinterpret_cast<int*>(buffer + address);
      *ptr     = thisrank;

      return count;
    }
  };

public:
  /// constructor
  LoadDiagnoser() : AsynchronousDiagnoser(2)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_output(data.curstep, config) == false)
      return;

    const int nc = data.cdims[3];

    // get parameters from json
    std::string prefix = config.value("prefix", "load");
    std::string path   = config.value("path", ".") + "/";

    // filename
    std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(data.curstep));
    std::string fn_json   = fn_prefix + ".json";
    std::string fn_data   = fn_prefix + ".data";

    size_t disp;
    json   dataset;

    open_file((path + fn_data).c_str(), &disp, "w");

    //
    // load
    //
    {
      const char name[]  = "load";
      const char desc[]  = "computational work load";
      const int  ndim    = 2;
      const int  dims[2] = {nc, App::Chunk::NumLoadMode};
      const int  size    = nc * App::Chunk::NumLoadMode * sizeof(float64);

      nixio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      launch(0, LoadPacker(), data, disp);
    }

    //
    // rank
    //
    {
      const char name[]  = "rank";
      const char desc[]  = "MPI rank";
      const int  ndim    = 1;
      const int  dims[1] = {nc};
      const int  size    = nc * sizeof(int);

      nixio::put_metadata(dataset, name, "i4", desc, disp, size, ndim, dims);
      launch(1, RankPacker(data.thisrank), data, disp);
    }

    if (is_completed() == true) {
      close_file();
    }

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

///
/// @brief Diagnoser for field
///
class FieldDiagnoser : public AsynchronousDiagnoser
{
protected:
  // data packer for electromagnetic field
  template <typename BasePacker>
  class FieldPacker : public BasePacker
  {
  public:
    using BasePacker::pack_field;

    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      return pack_field(data.uf, data, buffer, address);
    }
  };

  // data packer for current
  template <typename BasePacker>
  class CurrentPacker : public BasePacker
  {
  public:
    using BasePacker::pack_field;

    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      return pack_field(data.uj, data, buffer, address);
    }
  };

  // data packer for moment
  template <typename BasePacker>
  class MomentPacker : public BasePacker
  {
  public:
    using BasePacker::pack_field;

    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      return pack_field(data.um, data, buffer, address);
    }
  };

public:
  // constructor
  FieldDiagnoser() : AsynchronousDiagnoser(3)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_output(data.curstep, config) == false)
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

    size_t disp;
    json   dataset;

    open_file((path + fn_data).c_str(), &disp, "w");

    //
    // electromagnetic field
    //
    {
      const char name[]  = "uf";
      const char desc[]  = "electromagnetic field";
      const int  ndim    = 5;
      const int  dims[5] = {nc, nz, ny, nx, 6};
      const int  size    = nc * nz * ny * nx * 6 * sizeof(float64);

      nixio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      launch(0, FieldPacker<XtensorPacker3D>(), data, disp);
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

      nixio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      launch(1, CurrentPacker<XtensorPacker3D>(), data, disp);
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
      nixio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      launch(2, MomentPacker<XtensorPacker3D>(), data, disp);
    }

    if (is_completed() == true) {
      close_file();
    }

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

///
/// @brief Diagnoser for particle
///
class ParticleDiagnoser : public AsynchronousDiagnoser
{
protected:
  constexpr static int max_species = 10;

  // data packer for particle
  template <typename BasePacker>
  class ParticlePacker : public BasePacker
  {
  private:
    int index;

  public:
    using BasePacker::pack_particle;

    ParticlePacker(const int is) : index(is)
    {
    }

    template <typename Data>
    int operator()(Data data, uint8_t* buffer, int address)
    {
      return pack_particle(data.up[index], data, buffer, address);
    }
  };

public:
  // constructor
  ParticleDiagnoser() : AsynchronousDiagnoser(max_species)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_output(data.curstep, config) == false)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    assert(Ns <= max_species);

    // get parameters from json
    std::string prefix = config.value("prefix", "particle");
    std::string path   = config.value("path", ".") + "/";

    // filename
    std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(data.curstep));
    std::string fn_json   = fn_prefix + ".json";
    std::string fn_data   = fn_prefix + ".data";

    size_t disp;
    json   dataset;

    open_file((path + fn_data).c_str(), &disp, "w");

    //
    // for each particle
    //
    for (int is = 0; is < Ns; is++) {
      // write particles
      size_t disp0 = disp;
      launch(is, ParticlePacker<XtensorPacker3D>(is), data, disp);

      // meta data
      {
        std::string name = tfm::format("up%02d", is);
        std::string desc = tfm::format("particle species %02d", is);

        const int size    = ParticleType::get_particle_size();
        const int Np      = (disp - disp0) / size;
        const int ndim    = 2;
        const int dims[2] = {Np, ParticleType::Nc};

        nixio::put_metadata(dataset, name.c_str(), "f8", desc.c_str(), disp0, Np * size, ndim,
                            dims);
      }
    }

    if (is_completed() == true) {
      close_file();
    }

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

///
/// @brief Diagnoser
///
class Diagnoser
{
protected:
  HistoryDiagnoser  history;
  LoadDiagnoser     load;
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

    if (config["name"] == "load") {
      load(config, app, data);
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
