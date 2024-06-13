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

class BaseDiagnoser
{
protected:
  std::string name;
  std::string basedir;

public:
  // constructor
  BaseDiagnoser(std::string basedir, std::string name) : basedir(basedir), name(name)
  {
  }

  // check if the given key matches the name
  bool match(std::string key)
  {
    return key == name;
  }

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    int interval = config.value("interval", 1);
    int begin    = config.value("begin", 0);
    int end      = config.value("end", std::numeric_limits<int>::max());

    return (curstep >= begin) && (curstep < end) && ((curstep - begin) % interval == 0);
  }

  // check if the given step is the initial step
  bool is_initial_step(int curstep, json& config)
  {
    int begin = config.value("begin", 0);
    return curstep == begin;
  }

  // check if the parent directory of the given path exists
  bool make_sure_directory_exists(std::string path)
  {
    namespace fs = std::filesystem;

    fs::path filepath(path);
    fs::path dirpath = filepath.parent_path();

    if (fs::exists(dirpath) == true) {
      return true;
    }

    if (fs::create_directory(dirpath) == true) {
      return true;
    }

    ERROR << tfm::format("Failed to create directory: %s", path);

    return false;
  }

  // return formatted filename without directory
  std::string format_filename(json& config, std::string ext, std::string prefix, int step = -1)
  {
    namespace fs = std::filesystem;

    prefix = config.value("prefix", prefix);

    if (step >= 0) {
      prefix = tfm::format("%s_%s", prefix, nix::format_step(step));
    }

    return prefix + ext;
  }

  // return formatted filename
  std::string format_filename(json& config, std::string ext, std::string basedir, std::string path,
                              std::string prefix, int step = -1)
  {
    namespace fs = std::filesystem;

    path   = config.value("path", path);
    prefix = config.value("prefix", prefix);

    if (step >= 0) {
      prefix = tfm::format("%s_%s", prefix, nix::format_step(step));
    }

    return (fs::path(basedir) / path / prefix).string() + ext;
  }

  // calculate percentile assuming pre-sorted data
  template <typename T>
  float64 percentile(T& data, float64 p, bool is_sorted)
  {
    int     size  = data.size();
    int     index = p * (size - 1);
    float64 frac  = p * (size - 1) - index;

    if (is_sorted == false) {
      std::sort(data.begin(), data.end());
    }

    if (index >= 0 && index < size - 1) {
      // linear interpolation
      return data[index] * (1 - frac) + data[index + 1] * frac;
    } else {
      // error
      return -1;
    }
  }
};

class AsynchronousDiagnoser : public BaseDiagnoser
{
protected:
  MPI_File                 filehandle;
  bool                     is_opened;
  std::vector<Buffer>      buffer;
  std::vector<MPI_Request> request;

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    bool status    = BaseDiagnoser::require_diagnostic(curstep, config);
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
  AsynchronousDiagnoser(std::string basedir, std::string name, int size)
      : BaseDiagnoser(basedir, name), is_opened(false)
  {
    buffer.resize(size);
    request.resize(size);

    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
  }

  // open file
  void open_file(std::string filename, size_t* disp, const char* mode)
  {
    if (is_opened == false) {
      nixio::open_file(filename.c_str(), &filehandle, disp, mode);
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
    size_t bufsize = 0;

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
class HistoryDiagnoser : public BaseDiagnoser
{
public:
  // constructor
  HistoryDiagnoser(std::string basedir) : BaseDiagnoser(basedir, "history")
  {
  }

  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_diagnostic(data.curstep, config) == false)
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
      std::string filename = format_filename(config, ".txt", basedir, ".", "history");
      std::string msg      = "";

      // initial call
      if (is_initial_step(data.curstep, config) == true) {
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
        std::filesystem::remove(filename);
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
      if (make_sure_directory_exists(filename) == true) {
        std::ofstream ofs(filename, nix::text_append);
        ofs << msg << std::flush;
        ofs.close();
      }
    }
  }
};

///
/// @brief Diagnoser for resource usage
///
class ResourceDiagnoser : public BaseDiagnoser
{
protected:
  // calculate statistics
  template <typename T>
  auto statistics(T& data)
  {
    // sort
    std::sort(data.begin(), data.end());

    json stat      = {};
    stat["min"]    = data.front();
    stat["max"]    = data.back();
    stat["mean"]   = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    stat["quant1"] = percentile(data, 0.25, true);
    stat["quant2"] = percentile(data, 0.50, true);
    stat["quant3"] = percentile(data, 0.75, true);
    stat["size"]   = data.size();

    return stat;
  }

public:
  // constructor
  ResourceDiagnoser(std::string basedir) : BaseDiagnoser(basedir, "resource")
  {
  }

  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_diagnostic(data.curstep, config) == false)
      return;

    const float64        to_gb        = 1.0 / (1024 * 1024 * 1024);
    int                  local_chunk  = 0;
    float64              local_memory = 0;
    float64              local_load   = 0;
    float64              total_load   = 0;
    std::vector<float64> memoryvec;
    std::vector<float64> loadvec;
    std::vector<int>     node_chunk;
    std::vector<float64> node_memory;
    std::vector<float64> node_load;
    std::vector<int>     rank_chunk;
    std::vector<float64> rank_memory;
    std::vector<float64> rank_load;

    //
    // local resource usage
    //
    local_chunk = data.chunkvec.size();
    memoryvec.resize(local_chunk);
    loadvec.resize(local_chunk);
    for (int i = 0; i < local_chunk; i++) {
      memoryvec[i] = data.chunkvec[i]->get_size_byte() * to_gb;
      loadvec[i]   = data.chunkvec[i]->get_total_load();
    }
    local_memory = std::accumulate(memoryvec.begin(), memoryvec.end(), 0.0);
    local_load   = std::accumulate(loadvec.begin(), loadvec.end(), 0.0);

    // total load
    MPI_Reduce(&local_load, &total_load, 1, MPI_FLOAT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    //
    // resource usage
    //
    {
      MPI_Comm node_comm;
      MPI_Comm internode_comm;
      int      node_rank;
      int      node_color;
      int      internode_size;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, data.thisrank, MPI_INFO_NULL,
                          &node_comm);
      MPI_Comm_rank(node_comm, &node_rank);

      node_color = node_rank == 0 ? 0 : 1;
      MPI_Comm_split(MPI_COMM_WORLD, node_color, data.thisrank, &internode_comm);
      MPI_Comm_size(internode_comm, &internode_size);

      // for each node
      if (config.contains("node") == true) {
        node_chunk.resize(internode_size, 0);
        node_memory.resize(internode_size, 0);
        node_load.resize(internode_size, 0);

        // chunk
        {
          int  sum = 0;
          int* ptr = node_chunk.data();
          MPI_Reduce(&local_chunk, &sum, 1, MPI_INT, MPI_SUM, 0, node_comm);
          MPI_Gather(&sum, 1, MPI_INT, ptr, 1, MPI_INT, 0, internode_comm);
        }

        // memory
        {
          float64  sum = 0;
          float64* ptr = node_memory.data();
          MPI_Reduce(&local_memory, &sum, 1, MPI_FLOAT64_T, MPI_SUM, 0, node_comm);
          MPI_Gather(&sum, 1, MPI_FLOAT64_T, ptr, 1, MPI_FLOAT64_T, 0, internode_comm);
        }

        // load
        {
          float64  sum = 0;
          float64* ptr = node_load.data();
          MPI_Reduce(&local_load, &sum, 1, MPI_FLOAT64_T, MPI_SUM, 0, node_comm);
          MPI_Gather(&sum, 1, MPI_FLOAT64_T, ptr, 1, MPI_FLOAT64_T, 0, internode_comm);
          // normalize
          std::for_each(node_load.begin(), node_load.end(), [=](auto& x) { x /= total_load; });
        }
      }

      // for each rank
      if (config.contains("rank") == true) {
        rank_chunk.resize(data.nprocess, 0);
        rank_memory.resize(data.nprocess, 0);
        rank_load.resize(data.nprocess, 0);

        // chunk
        {
          int* ptr = rank_chunk.data();
          MPI_Gather(&local_chunk, 1, MPI_INT, ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // memory
        {
          float64* ptr = rank_memory.data();
          MPI_Gather(&local_memory, 1, MPI_FLOAT64_T, ptr, 1, MPI_FLOAT64_T, 0, MPI_COMM_WORLD);
        }

        // load
        {
          float64* ptr = rank_load.data();
          MPI_Gather(&local_load, 1, MPI_FLOAT64_T, ptr, 1, MPI_FLOAT64_T, 0, MPI_COMM_WORLD);
          // normalize
          std::for_each(rank_load.begin(), rank_load.end(), [=](auto& x) { x /= total_load; });
        }
      }

      MPI_Comm_free(&internode_comm);
      MPI_Comm_free(&node_comm);
    }

    // save to file
    {
      json result = {
          {"step", data.curstep},     {"rank", data.thisrank},      {"time", data.curtime},
          {"node_chunk", node_chunk}, {"node_memory", node_memory}, {"node_load", node_load},
          {"rank_chunk", rank_chunk}, {"rank_memory", rank_memory}, {"rank_load", rank_load}};

      savefile(config, result);
    }
  }

  void savefile(json& config, json& result)
  {
    // output from root
    if (result["rank"] == 0) {
      std::string filename = format_filename(config, ".msgpack", basedir, ".", "resource");

      json record    = {};
      record["step"] = result["step"];
      record["time"] = result["time"];

      // node
      if (config.contains("node") == true) {
        auto node_chunk  = result["node_chunk"].get<std::vector<int>>();
        auto node_memory = result["node_memory"].get<std::vector<float64>>();
        auto node_load   = result["node_load"].get<std::vector<float64>>();
        json chunk       = {};
        json memory      = {};
        json load        = {};

        if (config["node"] == "stats" || config["node"] == "full") {
          chunk["stats"]  = statistics(node_chunk);
          memory["stats"] = statistics(node_memory);
          load["stats"]   = statistics(node_load);
        }

        if (config["node"] == "full") {
          chunk["full"]  = node_chunk;
          memory["full"] = node_memory;
          load["full"]   = node_load;
        }

        record["node"] = {{"chunk", chunk}, {"memory", memory}, {"load", load}};
      }

      // rank
      if (config.contains("rank") == true) {
        auto rank_chunk  = result["rank_chunk"].get<std::vector<int>>();
        auto rank_memory = result["rank_memory"].get<std::vector<float64>>();
        auto rank_load   = result["rank_load"].get<std::vector<float64>>();
        json chunk       = {};
        json memory      = {};
        json load        = {};

        if (config["rank"] == "stats" || config["rank"] == "full") {
          chunk["stats"]  = statistics(rank_chunk);
          memory["stats"] = statistics(rank_memory);
          load["stats"]   = statistics(rank_load);
        }

        if (config["rank"] == "full") {
          chunk["full"]  = rank_chunk;
          memory["full"] = rank_memory;
          load["full"]   = rank_load;
        }

        record["rank"] = {{"chunk", chunk}, {"memory", memory}, {"load", load}};
      }

      // initial call
      if (is_initial_step(result["step"], config) == true) {
        std::filesystem::remove(filename);
      }

      // append to file
      if (make_sure_directory_exists(filename) == true) {
        std::ofstream             ofs(filename, nix::binary_append);
        std::vector<std::uint8_t> buffer = json::to_msgpack(record);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
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
    size_t operator()(Data data, uint8_t* buffer, int address)
    {
      auto& load = data.load;

      size_t count = sizeof(float64) * load.size() + address;

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
    size_t operator()(Data data, uint8_t* buffer, int address)
    {
      size_t count = sizeof(int) + address;

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
  LoadDiagnoser(std::string basedir) : AsynchronousDiagnoser(basedir, "load", 2)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_diagnostic(data.curstep, config) == false)
      return;

    const int nc = data.cdims[3];

    size_t      disp    = 0;
    json        dataset = {};
    std::string fn_data = format_filename(config, ".data", "load", data.curstep);
    std::string fn_json = format_filename(config, ".json", "load", data.curstep);
    std::string fn_data_with_path =
        format_filename(config, ".data", basedir, ".", "load", data.curstep);
    std::string fn_json_with_path =
        format_filename(config, ".json", basedir, ".", "load", data.curstep);

    if (data.thisrank == 0) {
      make_sure_directory_exists(fn_data_with_path);
    }

    open_file(fn_data_with_path, &disp, "w");

    //
    // load
    //
    {
      const char name[]  = "load";
      const char desc[]  = "computational work load";
      const int  ndim    = 2;
      const int  dims[2] = {nc, App::ChunkType::NumLoadMode};
      const int  size    = nc * App::ChunkType::NumLoadMode * sizeof(float64);

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
        std::ofstream ofs(fn_json_with_path);
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
    size_t operator()(Data data, uint8_t* buffer, int address)
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
    size_t operator()(Data data, uint8_t* buffer, int address)
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
    size_t operator()(Data data, uint8_t* buffer, int address)
    {
      return pack_field(data.um, data, buffer, address);
    }
  };

public:
  // constructor
  FieldDiagnoser(std::string basedir) : AsynchronousDiagnoser(basedir, "field", 3)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_diagnostic(data.curstep, config) == false)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    size_t      disp    = 0;
    json        dataset = {};
    std::string fn_data = format_filename(config, ".data", "field", data.curstep);
    std::string fn_json = format_filename(config, ".json", "field", data.curstep);
    std::string fn_data_with_path =
        format_filename(config, ".data", basedir, ".", "field", data.curstep);
    std::string fn_json_with_path =
        format_filename(config, ".json", basedir, ".", "field", data.curstep);

    if (data.thisrank == 0) {
      make_sure_directory_exists(fn_data_with_path);
    }

    open_file(fn_data_with_path, &disp, "w");

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
        std::ofstream ofs(fn_json_with_path);
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
    int     species;
    int     seed;
    float64 fraction;

  public:
    using BasePacker::pack_particle;

    ParticlePacker(int species, int seed = 0, float64 fraction = 1.0)
        : species(species), seed(seed), fraction(fraction)
    {
    }

    std::vector<int64_t> generate_random_index(int N, int M, int random_seed)
    {
      std::mt19937_64 engine(random_seed);

      std::vector<int64_t> index(N);
      std::iota(index.begin(), index.end(), 0);
      std::shuffle(index.begin(), index.end(), engine);
      index.resize(M);
      std::sort(index.begin(), index.end());

      return index;
    }

    template <typename Data>
    size_t operator()(Data data, uint8_t* buffer, int address)
    {
      if (fraction < 1.0) {
        const int N     = data.up[species]->Np;
        const int M     = N * fraction;
        auto      index = generate_random_index(N, M, seed);
        return pack_particle(data.up[species], index, buffer, address);
      } else {
        return pack_particle(data.up[species], buffer, address);
      }
    }
  };

public:
  // constructor
  ParticleDiagnoser(std::string basedir) : AsynchronousDiagnoser(basedir, "particle", max_species)
  {
  }

  // data packing functor
  template <typename App, typename Data>
  void operator()(json& config, App& app, Data& data)
  {
    if (require_diagnostic(data.curstep, config) == false)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    assert(Ns <= max_species);

    size_t      disp    = 0;
    json        dataset = {};
    std::string fn_data = format_filename(config, ".data", "particle", data.curstep);
    std::string fn_json = format_filename(config, ".json", "particle", data.curstep);
    std::string fn_data_with_path =
        format_filename(config, ".data", basedir, ".", "particle", data.curstep);
    std::string fn_json_with_path =
        format_filename(config, ".json", basedir, ".", "particle", data.curstep);

    if (data.thisrank == 0) {
      make_sure_directory_exists(fn_data_with_path);
    }

    open_file(fn_data_with_path, &disp, "w");

    //
    // for each particle
    //
    for (int is = 0; is < Ns; is++) {
      // write particles
      size_t  disp0    = disp;
      int     seed     = data.thisrank;
      float64 fraction = config.value("fraction", 0.01);
      launch(is, ParticlePacker<XtensorPacker3D>(is, seed, fraction), data, disp);

      // meta data
      {
        std::string name = tfm::format("up%02d", is);
        std::string desc = tfm::format("particle species %02d", is);

        const int size    = ParticleType::get_particle_size();
        const int Np      = (disp - disp0) / size;
        const int ndim    = 2;
        const int dims[2] = {Np, ParticleType::Nc};

        nixio::put_metadata(dataset, name, "f8", desc, disp0, Np * size, ndim, dims);
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
        std::ofstream ofs(fn_json_with_path);
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
  ResourceDiagnoser resource;
  LoadDiagnoser     load;
  FieldDiagnoser    field;
  ParticleDiagnoser particle;

public:
  Diagnoser(std::string basedir)
      : history(basedir), resource(basedir), load(basedir), field(basedir), particle(basedir)
  {
  }

  template <typename App, typename Data>
  void doit(json& config, App& app, Data& data)
  {
    if (config.contains("name") == false)
      return;

    if (history.match(config["name"]) == true) {
      history(config, app, data);
      return;
    }

    if (resource.match(config["name"]) == true) {
      resource(config, app, data);
      return;
    }

    if (load.match(config["name"]) == true) {
      load(config, app, data);
      return;
    }

    if (field.match(config["name"]) == true) {
      field(config, app, data);
      return;
    }

    if (particle.match(config["name"]) == true) {
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
