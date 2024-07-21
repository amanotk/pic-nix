// -*- C++ -*-
#ifndef _LOAD_DIAG_HPP_
#define _LOAD_DIAG_HPP_

#include "async.hpp"

///
/// @brief Diagnostic for computational work load
///
template <typename App, typename Data>
class LoadDiag : public AsyncDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::basedir;

  // data packer for load
  class LoadPacker
  {
  public:
    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
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

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
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
  LoadDiag(std::string basedir) : AsyncDiag<App, Data>(basedir, "load", 2)
  {
  }

  // data packing functor
  void operator()(json& config, App& app, Data& data) override
  {
    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    const int nc = data.cdims[3];

    size_t      disp    = 0;
    json        dataset = {};
    std::string fn_data = this->format_filename(config, ".data", "load", data.curstep);
    std::string fn_json = this->format_filename(config, ".json", "load", data.curstep);
    std::string fn_data_with_path =
        this->format_filename(config, ".data", basedir, ".", "load", data.curstep);
    std::string fn_json_with_path =
        this->format_filename(config, ".json", basedir, ".", "load", data.curstep);

    if (data.thisrank == 0) {
      this->make_sure_directory_exists(fn_data_with_path);
    }

    this->open_file(fn_data_with_path, &disp, "w");

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
      this->launch(0, LoadPacker(), data, disp);
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
      this->launch(1, RankPacker(data.thisrank), data, disp);
    }

    if (this->is_completed() == true) {
      this->close_file();
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
