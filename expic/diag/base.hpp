// -*- C++ -*-
#ifndef _BASE_DIAG_HPP_
#define _BASE_DIAG_HPP_

#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/nixio.hpp"
#include "nix/random.hpp"
#include "nix/xtensor_packer3d.hpp"

#include "mpi.h"

using namespace nix::typedefs;
using nix::json;
using nix::Buffer;
using nix::XtensorPacker3D;

class DiagInfo
{
public:
  MPI_Comm    intra_comm; // intra-node communicator
  MPI_Comm    inter_comm; // inter-node communicator
  int         world_rank; // rank
  int         world_size; // number of processes
  int         intra_size; // number of processes in the same node
  int         inter_size; // number of nodes
  int         intra_rank; // rank in the same node
  int         inter_rank; // rank of the node
  int         Ns;         // number of species
  std::string basedir;    // base directory
  std::string iomode;     // I/O mode

  DiagInfo(std::string basedir, std::string iomode, int Ns)
      : basedir(basedir), iomode(iomode), Ns(Ns)
  {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // intra-node communicator
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,
                        &intra_comm);
    MPI_Comm_size(intra_comm, &intra_size);
    MPI_Comm_rank(intra_comm, &intra_rank);

    // inter-node communicator
    MPI_Comm_split(MPI_COMM_WORLD, intra_rank != 0, world_rank, &inter_comm);
    {
      int buffer[2] = {0, 0};

      MPI_Comm_size(inter_comm, &buffer[0]);
      MPI_Comm_rank(inter_comm, &buffer[1]);

      MPI_Bcast(buffer, 2, MPI_INT, 0, intra_comm);

      inter_size = buffer[0];
      inter_rank = buffer[1];
    }
  }

  ~DiagInfo()
  {
    MPI_Comm_free(&intra_comm);
    MPI_Comm_free(&inter_comm);
  }
};

template <typename App, typename Data>
class BaseDiag
{
protected:
  std::string               name;
  std::shared_ptr<DiagInfo> info;

public:
  // constructor
  BaseDiag(std::string name, std::shared_ptr<DiagInfo> info) : name(name), info(info)
  {
    make_sure_directory_exists(format_dirname(""));
  }

  // destructor
  virtual ~BaseDiag() = default;

  // main operation
  virtual void operator()(json& config, App& app, Data& data) = 0;

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

    return (curstep >= begin) && (curstep <= end) && ((curstep - begin) % interval == 0);
  }

  bool is_json_required()
  {
    bool is_json_required_mpiio = info->iomode == "mpiio" && info->world_rank == 0;
    bool is_json_required_posix = info->iomode == "posix" && info->intra_rank == 0;

    return is_json_required_mpiio || is_json_required_posix;
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

    bool is_check_required_mpiio = info->iomode == "mpiio" && info->world_rank == 0;
    bool is_check_required_posix = info->iomode == "posix" && info->intra_rank == 0;

    if (is_check_required_mpiio || is_check_required_posix) {
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

    return true;
  }

  std::string get_prefix(json& config, std::string prefix)
  {
    return config.value("prefix", prefix);
  }

  std::string format_dirname(std::string prefix)
  {
    namespace fs = std::filesystem;

    std::string basedir = info->basedir;

    if (info->iomode == "mpiio") {
      fs::path dirname = fs::path(basedir) / fs::path(prefix) / "";
      return dirname.string();
    } else if (info->iomode == "posix") {
      std::string nodedir = tfm::format("node%06d", info->inter_rank);
      fs::path    dirname = fs::path(basedir) / fs::path(nodedir) / fs::path(prefix) / "";
      return dirname.string();
    } else {
      ERROR << tfm::format("Unknown I/O mode: %s", info->iomode);
      return "";
    }
  }

  std::string format_filename(std::string prefix, std::string ext, int step)
  {
    return prefix + nix::format_step(step) + ext;
  }

  int get_num_species()
  {
    return info->Ns;
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
