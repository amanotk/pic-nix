// -*- C++ -*-
#ifndef _DIAG_HPP_
#define _DIAG_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

class Diag
{
protected:
  class Info; // forward declaration
  using info_type = Info;

  /// class to hold MPI information
  class Info
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
    std::string basedir;    // base directory
    std::string iomode;     // I/O mode

    Info(std::string basedir, std::string iomode) : basedir(basedir), iomode(iomode)
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

    ~Info()
    {
      MPI_Comm_free(&intra_comm);
      MPI_Comm_free(&inter_comm);
    }
  };

  inline static std::shared_ptr<Info> info; // MPI information
  std::string                         name; // diagnostic name

public:
  // constructor
  Diag(std::string name) : name(name)
  {
    make_sure_directory_exists(format_dirname(""));
  }

  // main operation
  virtual void operator()(nix::json& config)
  {
    // do nothing
  }

  // check if the given key matches the name
  virtual bool match(std::string key)
  {
    return key == name;
  }

  // initialize static info
  static void initialize(std::string basedir, std::string iomode)
  {
    info = std::make_shared<Info>(basedir, iomode);
  }

  // finalize static info
  static void finalize()
  {
    if (info != nullptr) {
      info.reset();
    }
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
        nix::sync_directory(dirpath.string());
        return true;
      }

      ERROR << tfm::format("Failed to create directory: %s", dirpath.string());

      return false;
    }

    return true;
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
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
