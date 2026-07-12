// -*- C++ -*-
#ifndef _STATEHANDLER_HPP_
#define _STATEHANDLER_HPP_

#include "buffer.hpp"
#include "mpistream.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

static constexpr int default_max_file_per_dir = 1000;

///
/// @brief StateHandler class for managing simulation state persistence
///
/// The StateHandler class is responsible for saving and loading the simulation
/// state, including both application parameters and chunk vector data. It provides
/// methods to serialize the state into a message-packed binary format and to restore
/// the state from disk.
///
class StateHandler
{
protected:
  using Vector = std::vector<int64>;

  int         max_file_per_dir; ///< maximum number of files per directory
  std::string basedir;          ///< base directory

public:
  StateHandler(std::string basedir = "", int max_file_per_dir = default_max_file_per_dir)
      : basedir(basedir), max_file_per_dir(max_file_per_dir)
  {
  }

  template <typename PtrInterface>
  bool save(PtrInterface interface, std::string prefix)
  {
    write_status(interface, prefix, "in_progress", false);

    save_application(interface, prefix);
    save_chunkvec(interface, prefix);

    MPI_Barrier(MPI_COMM_WORLD);
    write_status(interface, prefix, "complete", true);

    return true;
  }

  template <typename PtrInterface>
  bool save_application(PtrInterface interface, std::string prefix)
  {
    auto data = interface->get_data();

    if (data.thisrank == 0) {
      json                      state  = interface->to_json();
      std::vector<std::uint8_t> buffer = json::to_msgpack(state);

      std::string   filename = get_path_with_basedir(prefix) + ".msgpack";
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      ofs.flush();
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return true;
  }

  template <typename PtrInterface>
  bool save_chunkvec(PtrInterface interface, std::string prefix)
  {
    auto data = interface->get_data();

    int thisrank = data.thisrank;
    int nprocess = data.nprocess;

    DEBUG2 << tfm::format("start saving chunkvec with prefix %s", prefix);

    std::string path = get_path_with_basedir(prefix);

    MpiStream::create_directory_tree(path, thisrank, nprocess, max_file_per_dir);
    std::string filename =
        MpiStream::get_filename(path, ".data", thisrank, nprocess, max_file_per_dir);

    {
      Vector id;
      Vector size;
      Vector offset;

      save_chunkvec_header(interface, filename, id, size, offset);
      save_chunkvec_content(interface, filename, id, size, offset);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    DEBUG2 << tfm::format("finish saving chunkvec with prefix %s", prefix);

    return true;
  }

  template <typename PtrInterface>
  bool load(PtrInterface interface, std::string prefix)
  {
    if (validate_status(interface, prefix) == false) {
      return false;
    }

    bool status = load_application(interface, prefix);
    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (status == false) {
      return false;
    }

    load_chunkvec(interface, prefix);

    return true;
  }

  template <typename PtrInterface>
  bool load_application(PtrInterface interface, std::string prefix)
  {
    std::string   filename = get_path_with_basedir(prefix) + ".msgpack";
    std::ifstream ifs(filename, std::ios::binary);

    json state  = json::from_msgpack(ifs);
    bool status = interface->from_json(state);

    ifs.close();

    return status;
  }

  template <typename PtrInterface>
  bool load_chunkvec(PtrInterface interface, std::string prefix)
  {
    auto data = interface->get_data();

    int thisrank = data.thisrank;
    int nprocess = data.nprocess;

    DEBUG2 << tfm::format("start loading chunkvec with prefix %s", prefix);

    std::string path = get_path_with_basedir(prefix, true);

    std::string filename =
        MpiStream::get_filename(path, ".data", thisrank, nprocess, max_file_per_dir);

    {
      Vector id;
      Vector size;
      Vector offset;

      load_chunkvec_header(interface, filename, id, size, offset);
      load_chunkvec_content(interface, filename, id, size, offset);
    }

    DEBUG2 << tfm::format("finish loading chunkvec with prefix %s", prefix);

    return true;
  }

  std::string get_status_filename(std::string prefix)
  {
    return get_path_with_basedir(prefix) + ".status.json";
  }

  template <typename PtrInterface>
  bool write_status(PtrInterface interface, std::string prefix, std::string status, bool atomic)
  {
    auto data = interface->get_data();

    if (data.thisrank == 0) {
      json payload = {{"status", status},
                      {"prefix", normalize_prefix(prefix)},
                      {"curstep", data.curstep},
                      {"curtime", data.curtime},
                      {"nprocess", data.nprocess},
                      {"timestamp", nix::wall_clock()},
                      {"application", prefix + ".msgpack"},
                      {"chunks", prefix}};

      write_status_payload(prefix, payload, atomic);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return true;
  }

  template <typename PtrInterface>
  bool validate_status(PtrInterface interface, std::string prefix)
  {
    auto data   = interface->get_data();
    bool status = true;

    if (data.thisrank == 0) {
      status = validate_status_payload(prefix, data.nprocess);
    }

    MPI_Bcast(&status, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    return status;
  }

protected:
  std::string normalize_prefix(const std::string& prefix) const
  {
    namespace fs = std::filesystem;
    return fs::weakly_canonical(fs::path(basedir) / fs::path(prefix)).string();
  }

  std::string get_path_with_basedir(std::string name, bool require_existence = false)
  {
    namespace fs = std::filesystem;

    fs::path base_path = fs::path(basedir);
    fs::path full_path = base_path / fs::path(name);

    if (require_existence == false) {
      return full_path.string();
    }

    if (fs::exists(full_path) == false) {
      // full_path should exist for loading
      // otherwise try to find it in the last directory of basedir
      fs::path last_dir = base_path;
      while (last_dir.filename() == "" || last_dir.filename() == ".") {
        last_dir = last_dir.parent_path();
      }

      full_path = last_dir.filename() / fs::path(name);
    }

    return full_path.string();
  }

  void write_status_payload(std::string prefix, json& payload, bool atomic)
  {
    namespace fs = std::filesystem;

    fs::path filename = get_status_filename(prefix);
    fs::path tmpname  = filename;
    tmpname += ".tmp";
    fs::path output = atomic ? tmpname : filename;

    std::ofstream ofs(output);
    ofs << payload.dump(2) << std::endl;
    ofs.flush();
    ofs.close();

    if (atomic == true) {
      fs::rename(tmpname, filename);
    }
  }

  bool validate_status_payload(std::string prefix, int nprocess)
  {
    std::string   filename = get_status_filename(prefix);
    std::ifstream ifs(filename);

    if (ifs.is_open() == false) {
      DEBUG0 << tfm::format("checkpoint status file not found: %s; treating as legacy checkpoint",
                            filename);
      return true;
    }

    json payload = json::parse(ifs, nullptr, false);
    if (payload.is_discarded()) {
      ERROR << tfm::format("failed to parse checkpoint status file: %s", filename);
      return false;
    }

    if (payload.contains("status") == false || payload["status"].is_string() == false ||
        payload["status"].get<std::string>() != "complete") {
      ERROR << tfm::format("checkpoint is not complete: %s", filename);
      return false;
    }

    if (payload.contains("prefix") == false || payload["prefix"].is_string() == false ||
        payload["prefix"].get<std::string>() != normalize_prefix(prefix)) {
      ERROR << tfm::format("checkpoint status prefix mismatch: %s", filename);
      return false;
    }

    if (payload.contains("nprocess") == false || payload["nprocess"].is_number_integer() == false ||
        payload["nprocess"].get<int>() != nprocess) {
      ERROR << tfm::format("checkpoint status nprocess mismatch: %s", filename);
      return false;
    }

    return true;
  }

  template <typename PtrInterface>
  bool save_chunkvec_header(PtrInterface interface, std::string filename, Vector& id, Vector& size,
                            Vector& offset)
  {
    auto data = interface->get_data();

    const int element_size = sizeof(Vector::value_type);

    int64 numchunk    = data.chunkvec.size();
    int64 header_size = (1 + numchunk * 3) * element_size;

    id.resize(numchunk);
    size.resize(numchunk);
    offset.resize(numchunk + 1, 0);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      id[i]   = data.chunkvec[i]->get_id();
      size[i] = data.chunkvec[i]->pack(nullptr, 0);
    }

    // calculate offset for each chunk
    std::partial_sum(size.begin(), size.end(), offset.begin() + 1);
    for (int i = 0; i < offset.size(); i++) {
      offset[i] += header_size;
    }

    // write to disk
    std::ofstream ofs(filename, std::ios::binary);

    ofs.write(reinterpret_cast<const char*>(&numchunk), element_size);
    ofs.write(reinterpret_cast<const char*>(id.data()), element_size * numchunk);
    ofs.write(reinterpret_cast<const char*>(size.data()), element_size * numchunk);
    ofs.write(reinterpret_cast<const char*>(offset.data()), element_size * numchunk);

    ofs.flush();
    ofs.close();

    return true;
  }

  template <typename PtrInterface>
  bool save_chunkvec_content(PtrInterface interface, std::string filename, Vector& id, Vector& size,
                             Vector& offset)
  {
    auto data = interface->get_data();

    Buffer buffer;
    buffer.resize(*std::max_element(size.begin(), size.end()));

    std::ofstream ofs(filename, std::ios::binary | std::ios::app);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto& chunk = data.chunkvec[i];

      if (size[i] == chunk->pack(buffer.get(), 0) && id[i] == chunk->get_id()) {
        ofs.seekp(offset[i], std::ios::beg);
        ofs.write(reinterpret_cast<const char*>(buffer.get()), size[i]);
      } else {
        ERROR << tfm::format("Error in writing Chunk ID %08d", id[i]);
      }
    }

    ofs.close();
    ofs.close();

    return true;
  }

  template <typename PtrInterface>
  bool load_chunkvec_header(PtrInterface interface, std::string filename, Vector& id, Vector& size,
                            Vector& offset)
  {
    const int element_size = sizeof(Vector::value_type);

    int64 numchunk = 0;

    std::ifstream ifs(filename, std::ios::binary);

    ifs.read(reinterpret_cast<char*>(&numchunk), element_size);

    id.resize(numchunk);
    size.resize(numchunk);
    offset.resize(numchunk);

    ifs.read(reinterpret_cast<char*>(id.data()), element_size * numchunk);
    ifs.read(reinterpret_cast<char*>(size.data()), element_size * numchunk);
    ifs.read(reinterpret_cast<char*>(offset.data()), element_size * numchunk);

    ifs.close();

    return true;
  }

  template <typename PtrInterface>
  bool load_chunkvec_content(PtrInterface interface, std::string filename, Vector& id, Vector& size,
                             Vector& offset)
  {
    auto data = interface->get_data();

    Buffer buffer;
    buffer.resize(*std::max_element(size.begin(), size.end()));

    // local dimensions
    Bool3D has_dim{
        (data.ndims[0] == 1 && data.cdims[0] == 1) ? false : true,
        (data.ndims[1] == 1 && data.cdims[1] == 1) ? false : true,
        (data.ndims[2] == 1 && data.cdims[2] == 1) ? false : true,
    };
    Dims3D dims{
        data.ndims[0] / data.cdims[0],
        data.ndims[1] / data.cdims[1],
        data.ndims[2] / data.cdims[2],
    };

    // clear
    data.chunkvec.resize(0);
    data.chunkvec.shrink_to_fit();

    std::ifstream ifs(filename, std::ios::binary);

    // read data
    for (int i = 0; i < id.size(); i++) {
      auto chunk = interface->create_chunk(dims, has_dim, 0);

      ifs.seekg(offset[i], std::ios::beg);
      ifs.read(reinterpret_cast<char*>(buffer.get()), size[i]);

      // restore
      if (size[i] == chunk->unpack(buffer.get(), 0) && id[i] == chunk->get_id()) {
        data.chunkvec.push_back(std::move(chunk));
      } else {
        ERROR << tfm::format("Error in reading Chunk ID %08d", id[i]);
      }
    }

    ifs.close();

    return true;
  }
};

NIX_NAMESPACE_END

#endif
