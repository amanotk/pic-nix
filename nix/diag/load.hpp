// -*- C++ -*-
#ifndef _DIAG_LOAD_HPP_
#define _DIAG_LOAD_HPP_

#include "chunk.hpp"
#include "diag.hpp"
#include "diag/metadata.hpp"
#include "diag/parallel.hpp"
#include "nixio.hpp"

NIX_NAMESPACE_BEGIN

template <typename BaseDiag, typename Packer>
class LoadDiag : public ParallelDiag<BaseDiag, Packer>
{
protected:
  // data packer for load
  class LoadPacker : public Packer
  {
  public:
    using chunk_data_type = typename Packer::chunk_data_type;

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
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
  class RankPacker : public Packer
  {
  private:
    int thisrank;

  public:
    using chunk_data_type = typename Packer::chunk_data_type;

    RankPacker(int rank) : thisrank(rank)
    {
    }

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
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
  LoadDiag(typename BaseDiag::PtrInterface interface)
      : ParallelDiag<BaseDiag, Packer>(diag_name, interface)
  {
  }

  // data packing functor
  void operator()(json& config) override
  {
    auto data = this->interface->get_data();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    size_t      disp    = 0;
    json        dataset = {};
    std::string prefix  = this->get_prefix(config, "load");
    std::string dirname = this->format_dirname(prefix);
    std::string fn_data = this->format_filename("", ".data", data.curstep);
    std::string fn_json = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    //
    // load
    //
    {
      // data
      auto   packer = LoadPacker();
      size_t disp0  = disp;
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);

      // determine load vector size from first chunk
      size_t load_size = 0;
      if (data.chunkvec.size() > 0) {
        load_size = data.chunkvec[0]->get_load().size();
      }

      // metadata
      const char name[]  = "load";
      const char desc[]  = "computational work load";
      int        ndim    = 2;
      int        dims[2] = {0, static_cast<int>(load_size)};

      if (load_size > 0) {
        size_t size = load_size * sizeof(float64);
        dims[0]     = static_cast<int>(nbyte / size);
        nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
      } else {
        nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
      }
    }

    //
    // rank
    //
    {
      // data
      auto   packer = RankPacker(data.thisrank);
      size_t disp0  = disp;
      size_t size   = sizeof(int);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "rank";
      const char desc[]  = "MPI rank";
      int        ndim    = 1;
      int        dims[1] = {nc};
      nixio::put_metadata(dataset, name, "i4", desc, disp0, nbyte, ndim, dims);
    }

    if (this->is_completed() == true) {
      this->close_file();
    }

    //
    // output json file
    //
    auto chunk_id_range = this->get_chunk_id_range(data);

    if (this->is_json_required() == true) {
      json root;

      // meta data
      root["meta"] = make_metadata(fn_data, data.curtime, data.curstep, chunk_id_range);
      // dataset
      root["dataset"] = dataset;

      std::ofstream ofs(dirname + fn_json);
      ofs << std::setw(2) << root;
      ofs.flush();
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  static constexpr const char* diag_name = "load";
};

NIX_NAMESPACE_END

#endif
