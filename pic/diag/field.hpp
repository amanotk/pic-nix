// -*- C++ -*-
#ifndef _FIELD_DIAG_HPP_
#define _FIELD_DIAG_HPP_

#include "parallel.hpp"

///
/// @brief Diagnostic for field
///
class FieldDiag : public ParallelDiag
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

public:
  // constructor
  FieldDiag(PtrInterface interface) : ParallelDiag(diag_name, interface)
  {
  }

  // data packing functor
  void operator()(json& config) override
  {
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
      size_t nbyte  = this->queue(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "uf";
      const char desc[]  = "electromagnetic field";
      int        ndim    = 5;
      int        dims[5] = {nc, nz, ny, nx, 6};
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
      size_t nbyte  = this->queue(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "um";
      const char desc[]  = "moment";
      int        ndim    = 6;
      int        dims[6] = {nc, nz, ny, nx, Ns, 14};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    return dataset;
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
