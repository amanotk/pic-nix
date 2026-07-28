// -*- C++ -*-
#ifndef _HYBRID_FIELD_DIAG_HPP_
#define _HYBRID_FIELD_DIAG_HPP_

#include "hybrid/diag/chunk_writer.hpp"
#include "hybrid/diag/hybrid_diag.hpp"

#include "nix/diag/metadata.hpp"

#include <fstream>
#include <iomanip>

namespace hybrid
{
class FieldDiag : public HybridChunkDiagWriter
{
public:
  static constexpr const char* diag_name = "field";

protected:
  class FieldCellPacker : public HybridPacker
  {
  public:
    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      return pack_array_raw(data.field_cell, data, buffer, address);
    }
  };

  class FluidPacker : public HybridPacker
  {
  public:
    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      return pack_array_raw(data.fluid, data, buffer, address);
    }
  };

  class MomentPacker : public HybridPacker
  {
  public:
    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      return pack_array_raw(data.moment_kinetic, data, buffer, address);
    }
  };

public:
  FieldDiag(PtrInterface interface) : HybridChunkDiagWriter(diag_name, interface)
  {
  }

  void operator()(nix::json& config) override
  {
    auto data = interface->get_data();
    auto Ns   = interface->get_num_species();

    if (require_diagnostic(data.curstep, config) == false)
      return;

    size_t      disp    = 0;
    json        dataset = {};
    std::string prefix  = this->get_prefix(config, "field");
    std::string dirname = this->format_dirname(prefix);
    std::string fn_data = this->format_filename("", ".data", data.curstep);
    std::string fn_json = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    const int nz   = data.ndims[0] / data.cdims[0];
    const int ny   = data.ndims[1] / data.cdims[1];
    const int nx   = data.ndims[2] / data.cdims[2];
    const int nc6  = static_cast<int>(num_field_components);
    const int nc10 = static_cast<int>(num_fluid_components);
    const int ncM  = static_cast<int>(num_moment_components);

    //
    // electromagnetic field (field_cell)
    //
    {
      auto   packer = FieldCellPacker();
      size_t disp0  = disp;
      size_t esize  = static_cast<size_t>(nz) * ny * nx * nc6 * sizeof(nix::float64);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / esize);

      const char name[]  = "field_cell";
      const char desc[]  = "cell-centered electromagnetic field";
      int        ndim    = 5;
      int        dims[5] = {nc, nz, ny, nx, nc6};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    //
    // fluid variables
    //
    {
      auto   packer = FluidPacker();
      size_t disp0  = disp;
      size_t esize  = static_cast<size_t>(nz) * ny * nx * nc10 * sizeof(nix::float64);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / esize);

      const char name[]  = "fluid";
      const char desc[]  = "ion/electron fluid variables";
      int        ndim    = 5;
      int        dims[5] = {nc, nz, ny, nx, nc10};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    //
    // kinetic moment
    //
    {
      auto   packer = MomentPacker();
      size_t disp0  = disp;
      size_t esize  = static_cast<size_t>(nz) * ny * nx * Ns * ncM * sizeof(nix::float64);
      size_t nbyte  = this->write_packed_chunks(packer, data, disp);
      int    nc     = static_cast<int>(nbyte / esize);

      const char name[]  = "moment";
      const char desc[]  = "kinetic moments";
      int        ndim    = 6;
      int        dims[6] = {nc, nz, ny, nx, Ns, ncM};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
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

      root["meta"]    = nix::make_metadata(fn_data, data.curtime, data.curstep, chunk_id_range);
      root["dataset"] = dataset;

      std::ofstream ofs(dirname + fn_json);
      ofs << std::setw(2) << root;
      ofs.flush();
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};
} // namespace hybrid

#endif
