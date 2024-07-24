// -*- C++ -*-
#ifndef _FIELD_DIAG_HPP_
#define _FIELD_DIAG_HPP_

#include "async.hpp"

///
/// @brief Diagnostic for field
///
template <typename App, typename Data>
class FieldDiag : public AsyncDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::info;

  // data packer for electromagnetic field
  template <typename BasePacker>
  class FieldPacker : public BasePacker
  {
  public:
    using BasePacker::pack_field;

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
    {
      return pack_field(data.uf, data, buffer, address);
    }
  };

  // data packer for moment
  template <typename BasePacker>
  class MomentPacker : public BasePacker
  {
  public:
    using BasePacker::pack_field;

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
    {
      return pack_field(data.um, data, buffer, address);
    }
  };

public:
  // constructor
  FieldDiag(std::shared_ptr<DiagInfo> info) : AsyncDiag<App, Data>("field", info, 2)
  {
  }

  // data packing functor
  void operator()(json& config, App& app, Data& data) override
  {
    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    const int nz = data.ndims[0] / data.cdims[0];
    const int ny = data.ndims[1] / data.cdims[1];
    const int nx = data.ndims[2] / data.cdims[2];
    const int Ns = this->get_num_species();

    size_t      disp    = 0;
    json        dataset = {};
    std::string prefix  = this->get_prefix(config, "field");
    std::string dirname = this->format_dirname(prefix);
    std::string fn_data = this->format_filename("", ".data", data.curstep);
    std::string fn_json = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    //
    // electromagnetic field
    //
    {
      // data
      auto   packer = FieldPacker<XtensorPacker3D>();
      size_t disp0  = disp;
      size_t size   = nz * ny * nx * 6 * sizeof(float64);
      size_t nbyte  = this->launch(0, packer, data, disp);
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
    app.calculate_moment();
    {
      // data
      auto   packer = MomentPacker<XtensorPacker3D>();
      size_t disp0  = disp;
      size_t size   = nz * ny * nx * Ns * 11 * sizeof(float64);
      size_t nbyte  = this->launch(1, packer, data, disp);
      int    nc     = static_cast<int>(nbyte / size);

      // metadata
      const char name[]  = "um";
      const char desc[]  = "moment";
      int        ndim    = 6;
      int        dims[6] = {nc, nz, ny, nx, Ns, 11};
      nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
    }

    if (this->is_completed() == true) {
      this->close_file();
    }

    //
    // output json file
    //
    if (this->is_json_required() == true) {
      json root;

      // meta data
      root["meta"] = {{"endian", nix::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"order", 1},
                      {"time", data.curtime},
                      {"step", data.curstep}};
      // dataset
      root["dataset"] = dataset;

      std::ofstream ofs(dirname + fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
