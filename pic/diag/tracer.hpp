// -*- C++ -*-
#ifndef _TRACER_DIAG_HPP_
#define _TRACER_DIAG_HPP_

#include "chunk_writer.hpp"

///
/// @brief Diagnostic for tracer
///
class TracerDiag : public PicChunkDiagWriter
{
public:
  static constexpr const char* diag_name = "tracer";

protected:
  // data packer for particle
  class TracerPacker : public PicPacker
  {
  private:
    int species;
    int seed;

  public:
    TracerPacker(int species, int seed = 0) : species(species), seed(seed)
    {
    }

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      return pack_tracer(data.up[species], buffer, address);
    }
  };

public:
  // constructor
  TracerDiag(PtrInterface interface) : PicChunkDiagWriter(diag_name, interface)
  {
  }

  // data packing functor
  virtual void operator()(json& config) override
  {
    auto data = interface->get_data();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    size_t      disp    = 0;
    json        dataset = {};
    std::string prefix  = this->get_prefix(config, "tracer");
    std::string dirname = this->format_dirname(prefix);
    std::string fn_data = this->format_filename("", ".data", data.curstep);
    std::string fn_json = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    {
      // write particles
      int    species = config.value("species", 0);
      int    seed    = data.thisrank;
      auto   packer  = TracerPacker(species, seed);
      size_t disp0   = disp;
      size_t nbyte   = this->write_packed_chunks(packer, data, disp);

      // meta data
      {
        std::string name = fmt::format("up{:02d}", species);
        std::string desc = fmt::format("tracer particle species {:02d}", species);

        const int size    = ParticleType::get_particle_size();
        const int Np      = nbyte / size;
        const int ndim    = 2;
        const int dims[2] = {Np, ParticleType::Nc};

        nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
      }
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
                      {"layout", nix::ARRAY_LAYOUT},
                      {"time", data.curtime},
                      {"step", data.curstep}};
      // dataset
      root["dataset"] = dataset;

      std::ofstream ofs(dirname + fn_json);
      ofs << std::setw(2) << root;
      ofs.flush();
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

#endif
