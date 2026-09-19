// -*- C++ -*-
#ifndef _DIAG_LOAD_HPP_
#define _DIAG_LOAD_HPP_

#include "chunk.hpp"
#include "diag.hpp"
#include "diag/chunk_writer.hpp"
#include "diag/metadata.hpp"
#include "nixio.hpp"

#if PICNIX_ENABLE_ADIOS2
#include "../../pic/diag/adios2_writer.hpp"
#include <map>
#endif

NIX_NAMESPACE_BEGIN

template <typename BaseDiag, typename Packer>
class LoadDiag : public ChunkDiagWriter<BaseDiag, Packer>
{
protected:
  using chunk_type = typename BaseDiag::chunk_type;

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

#if PICNIX_ENABLE_ADIOS2
  struct Adios2State {
    std::unique_ptr<nix::Adios2Writer> writer;
    bool                               variables_defined = false;
  };

  std::map<std::string, Adios2State> adios2_state;

  void write_adios2(json& config)
  {
    auto data = this->interface->get_data();
    if (this->require_diagnostic(data.curstep, config) == false) {
      return;
    }

    const std::string prefix      = this->get_prefix(config, "load");
    auto&             state       = adios2_state[prefix];
    const int         local_count = data.chunkvec.size();

    int local_min      = local_count > 0 ? std::numeric_limits<int>::max() : 0;
    int local_max      = local_count > 0 ? std::numeric_limits<int>::min() : -1;
    int local_load_min = std::numeric_limits<int>::max();
    int local_load_max = local_count > 0 ? 0 : std::numeric_limits<int>::min();
    for (int i = 0; i < local_count; i++) {
      auto chunk     = static_cast<chunk_type*>(data.chunkvec[i].get());
      auto load      = chunk->get_load();
      local_min      = std::min(local_min, chunk->get_id());
      local_max      = std::max(local_max, chunk->get_id());
      local_load_min = std::min(local_load_min, static_cast<int>(load.size()));
      local_load_max = std::max(local_load_max, static_cast<int>(load.size()));
    }

    int valid           = local_count == 0 || local_max - local_min + 1 == local_count ? 1 : 0;
    int global_min      = 0;
    int global_max      = 0;
    int global_count    = 0;
    int global_load_min = 0;
    int global_load_max = 0;
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_load_min, &global_load_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_load_max, &global_load_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (valid == 0 || global_min != 0 || global_max != data.cdims[3] - 1 ||
        global_count != data.cdims[3] || global_load_min == std::numeric_limits<int>::max() ||
        global_load_min != global_load_max) {
      ERROR << "ADIOS2 load output requires contiguous chunks with uniform non-empty load vectors";
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const int                 load_size = global_load_min;
    std::vector<float64>      load_values(static_cast<size_t>(local_count) * load_size);
    std::vector<std::int32_t> rank_values(static_cast<size_t>(local_count), data.thisrank);
    for (int i = 0; i < local_count; i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      auto load  = chunk->get_load();
      std::copy(load.begin(), load.end(), load_values.begin() + i * load_size);
    }

    if (state.writer == nullptr) {
      state.writer = std::make_unique<nix::Adios2Writer>(this->info);
      state.writer->initialize("load", prefix, this->interface->get_configuration());

      const size_t chunk_count = static_cast<size_t>(data.cdims[3]);
      state.writer->define_global_double(
          "load", {chunk_count, static_cast<size_t>(load_size)},
          {static_cast<size_t>(local_min), 0},
          {static_cast<size_t>(local_count), static_cast<size_t>(load_size)});
      state.writer->define_global_int32("rank", {chunk_count}, {static_cast<size_t>(local_min)},
                                        {static_cast<size_t>(local_count)});
      state.writer->open();
      state.variables_defined = true;
    }

    state.writer->begin_step(static_cast<std::int64_t>(data.curstep), data.curtime);
    state.writer->put_global_double(
        "load", {static_cast<size_t>(local_min), 0},
        {static_cast<size_t>(local_count), static_cast<size_t>(load_size)}, load_values.data(),
        load_values.size());
    state.writer->put_global_int32("rank", {static_cast<size_t>(local_min)},
                                   {static_cast<size_t>(local_count)}, rank_values.data(),
                                   rank_values.size());
    state.writer->end_step();
  }
#endif

public:
  /// constructor
  LoadDiag(typename BaseDiag::PtrInterface interface)
      : ChunkDiagWriter<BaseDiag, Packer>(diag_name, interface)
  {
  }

  void shutdown()
  {
#if PICNIX_ENABLE_ADIOS2
    for (auto& [prefix, state] : adios2_state) {
      if (state.writer != nullptr) {
        state.writer->close();
      }
    }
#endif
  }

  // data packing functor
  void operator()(json& config) override
  {
#if PICNIX_ENABLE_ADIOS2
    if (this->info->iomode == "adios2") {
      write_adios2(config);
      return;
    }
#endif

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
      int64      dims[2] = {0, static_cast<int64>(load_size)};

      if (load_size > 0) {
        size_t size = load_size * sizeof(float64);
        dims[0]     = static_cast<int64>(nbyte / size);
      }
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
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
      int64      dims[1] = {nc};
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
