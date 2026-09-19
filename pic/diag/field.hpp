// -*- C++ -*-
#ifndef _FIELD_DIAG_HPP_
#define _FIELD_DIAG_HPP_

#include "chunk_writer.hpp"

#if PICNIX_ENABLE_ADIOS2
#include "adios2_writer.hpp"
#include <map>
#endif

///
/// @brief Diagnostic for field
///
class FieldDiag : public PicChunkDiagWriter
{
public:
  static constexpr const char* diag_name = "field";

protected:
  // data packer for electromagnetic field
  class FieldPacker : public PicPacker
  {
  private:
    int decimate;

  public:
    FieldPacker(int decimate = -1) : decimate(decimate)
    {
    }

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      if (decimate < 0) {
        return pack_array_raw(data.uf, data, buffer, address);
      } else {
        return pack_field(data.uf, data, decimate, buffer, address);
      }
    }
  };

  // data packer for moment
  class MomentPacker : public PicPacker
  {
  private:
    int decimate;

  public:
    MomentPacker(int decimate = -1) : decimate(decimate)
    {
    }

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      if (decimate < 0) {
        return pack_array_raw(data.um, data, buffer, address);
      } else {
        return pack_moment(data.um, data, decimate, buffer, address);
      }
    }
  };

#if PICNIX_ENABLE_ADIOS2
  struct Adios2State {
    std::unique_ptr<nix::Adios2Writer> writer;
    bool                               variables_defined = false;
  };

  std::map<std::string, Adios2State> adios2_state;

  template <typename Packer>
  std::vector<float64> pack_adios2_chunks(Packer& packer, data_type& data)
  {
    size_t nbyte = 0;
    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      nbyte += packer(chunk->get_internal_data(), nullptr, 0);
    }

    std::vector<float64> values(nbyte / sizeof(float64));
    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      address =
          packer(chunk->get_internal_data(), reinterpret_cast<uint8_t*>(values.data()), address);
    }
    return values;
  }

  void write_adios2(json& config)
  {
    auto data = interface->get_data();
    if (this->require_diagnostic(data.curstep, config) == false) {
      return;
    }

    const int         decimate = config.value("decimate", 1);
    const std::string prefix   = this->get_prefix(config, "field");
    const int         Ns       = interface->get_num_species();
    auto&             state    = adios2_state[prefix];

    const int local_count = data.chunkvec.size();
    int       local_min   = local_count > 0 ? std::numeric_limits<int>::max() : 0;
    int       local_max   = local_count > 0 ? std::numeric_limits<int>::min() : -1;
    for (int i = 0; i < local_count; i++) {
      const int id = data.chunkvec[i]->get_id();
      local_min    = std::min(local_min, id);
      local_max    = std::max(local_max, id);
    }

    const bool local_contiguous = local_count == 0 || local_max - local_min + 1 == local_count;
    int        valid            = local_contiguous ? 1 : 0;
    int        global_min       = 0;
    int        global_max       = 0;
    int        global_count     = 0;
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (valid == 0 || global_min != 0 || global_max != data.cdims[3] - 1 ||
        global_count != data.cdims[3]) {
      ERROR << "ADIOS2 field output requires contiguous global chunk ownership";
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const int nz = this->calc_decimated_size(data.ndims[0] / data.cdims[0], decimate);
    const int ny = this->calc_decimated_size(data.ndims[1] / data.cdims[1], decimate);
    const int nx = this->calc_decimated_size(data.ndims[2] / data.cdims[2], decimate);

    FieldPacker field_packer(decimate);
    auto        uf = pack_adios2_chunks(field_packer, data);

    interface->calculate_moment();
    MomentPacker moment_packer(decimate);
    auto         um = pack_adios2_chunks(moment_packer, data);

    if (state.writer == nullptr) {
      state.writer = std::make_unique<nix::Adios2Writer>(this->info);
      state.writer->initialize("field", prefix, interface->get_configuration());

      const size_t chunk_count = static_cast<size_t>(data.cdims[3]);
      state.writer->define_global_double("uf",
                                         {chunk_count, static_cast<size_t>(nz),
                                          static_cast<size_t>(ny), static_cast<size_t>(nx), 6},
                                         {static_cast<size_t>(local_min), 0, 0, 0, 0},
                                         {static_cast<size_t>(local_count), static_cast<size_t>(nz),
                                          static_cast<size_t>(ny), static_cast<size_t>(nx), 6});
      state.writer->define_global_double(
          "um",
          {chunk_count, static_cast<size_t>(nz), static_cast<size_t>(ny), static_cast<size_t>(nx),
           static_cast<size_t>(Ns), 14},
          {static_cast<size_t>(local_min), 0, 0, 0, 0, 0},
          {static_cast<size_t>(local_count), static_cast<size_t>(nz), static_cast<size_t>(ny),
           static_cast<size_t>(nx), static_cast<size_t>(Ns), 14});
      state.writer->open();
      state.variables_defined = true;
    }

    if (state.variables_defined == false) {
      throw std::logic_error("ADIOS2 field variables were not initialized");
    }

    state.writer->begin_step(static_cast<std::int64_t>(data.curstep), data.curtime);
    state.writer->put_global_double("uf", {static_cast<size_t>(local_min), 0, 0, 0, 0},
                                    {static_cast<size_t>(local_count), static_cast<size_t>(nz),
                                     static_cast<size_t>(ny), static_cast<size_t>(nx), 6},
                                    uf.data(), uf.size());
    state.writer->put_global_double("um", {static_cast<size_t>(local_min), 0, 0, 0, 0, 0},
                                    {static_cast<size_t>(local_count), static_cast<size_t>(nz),
                                     static_cast<size_t>(ny), static_cast<size_t>(nx),
                                     static_cast<size_t>(Ns), 14},
                                    um.data(), um.size());
    state.writer->end_step();
  }
#endif

public:
  // constructor
  FieldDiag(PtrInterface interface) : PicChunkDiagWriter(diag_name, interface)
  {
  }

  void shutdown() override
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

    auto data = interface->get_data();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    int         decimate = config.value("decimate", 1); // default 1
    size_t      disp     = 0;
    std::string prefix   = this->get_prefix(config, "field");
    std::string dirname  = this->format_dirname(prefix);
    std::string fn_data  = this->format_filename("", ".data", data.curstep);
    std::string fn_json  = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    json dataset = write_decimated_data(config, decimate, disp);

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
      root["meta"] = {{"endian", nix::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"layout", nix::ARRAY_LAYOUT},
                      {"decimate", decimate},
                      {"time", data.curtime},
                      {"step", data.curstep},
                      {"chunk_id_range", chunk_id_range}};
      // dataset
      root["dataset"] = dataset;

      std::ofstream ofs(dirname + fn_json);
      ofs << std::setw(2) << root;
      ofs.flush();
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // calculate decimated array size
  int calc_decimated_size(int size, int decimate)
  {
    if (size <= decimate) {
      // collapse to size 1
      return 1;
    } else if (decimate <= 0 || size % decimate != 0) {
      // invalid input, fallback to original size
      return size;
    }

    return size / decimate;
  }

  // write decimated data
  json write_decimated_data(json& config, int decimate, size_t& disp)
  {
    auto data = interface->get_data();
    auto Ns   = interface->get_num_species();

    const int nz = calc_decimated_size(data.ndims[0] / data.cdims[0], decimate);
    const int ny = calc_decimated_size(data.ndims[1] / data.cdims[1], decimate);
    const int nx = calc_decimated_size(data.ndims[2] / data.cdims[2], decimate);

    json dataset = {};

    //
    // electromagnetic field
    //
    {
      // data
      auto   packer = FieldPacker(decimate);
      size_t disp0  = disp;
      size_t size   = nz * ny * nx * 6 * sizeof(float64);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "uf";
      const char desc[]  = "electromagnetic field";
      int        ndim    = 5;
      int64      dims[5] = {nc, nz, ny, nx, 6};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    //
    // moment
    //
    interface->calculate_moment();
    {
      // data
      auto   packer = MomentPacker(decimate);
      size_t disp0  = disp;
      size_t size   = nz * ny * nx * Ns * 14 * sizeof(float64);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "um";
      const char desc[]  = "moment";
      int        ndim    = 6;
      int64      dims[6] = {nc, nz, ny, nx, Ns, 14};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    return dataset;
  }
};

#endif
