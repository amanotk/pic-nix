// -*- C++ -*-
#ifndef _TRACER_DIAG_HPP_
#define _TRACER_DIAG_HPP_

#include "chunk_writer.hpp"

#if PICNIX_ENABLE_ADIOS2
#include "adios2_writer.hpp"
#include <map>
#endif

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

#if PICNIX_ENABLE_ADIOS2
  struct Adios2State {
    std::unique_ptr<nix::Adios2Writer> writer;
    bool                               variables_defined = false;
  };

  std::map<std::string, Adios2State> adios2_state;

  void write_adios2(json& config)
  {
    auto data = interface->get_data();
    if (this->require_diagnostic(data.curstep, config) == false) {
      return;
    }

    const int                  species = config.value("species", 0);
    const std::string          prefix  = this->get_prefix(config, "tracer");
    const std::string          name    = fmt::format("up{:02d}", species);
    const size_t               width   = ParticleType::Nc - 1;
    auto&                      state   = adios2_state[prefix];
    std::vector<float64>       values;
    std::vector<std::uint64_t> ids;

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto         chunk        = static_cast<chunk_type*>(data.chunkvec[i].get());
      auto         cdata        = chunk->get_internal_data();
      auto&        particle     = cdata.up[species];
      const size_t active_count = particle->get_Np_active();
      values.reserve(values.size() + active_count * width);
      ids.reserve(ids.size() + active_count);
      for (size_t ip = 0; ip < active_count; ip++) {
        std::int64_t id = 0;
        std::memcpy(&id, &particle->xu(ip, ParticleType::Nc - 1), sizeof(id));
        if (id >= 0) {
          continue;
        }

        for (size_t ic = 0; ic < width; ic++) {
          values.push_back(particle->xu(ip, ic));
        }
        std::uint64_t raw_id = 0;
        std::memcpy(&raw_id, &particle->xu(ip, ParticleType::Nc - 1), sizeof(raw_id));
        ids.push_back(raw_id);
      }
    }

    if (state.writer == nullptr) {
      state.writer = std::make_unique<nix::Adios2Writer>(this->info);
      state.writer->initialize("tracer", prefix, interface->get_configuration());
      state.writer->define_joined_double(name, width, ids.size());
      state.writer->define_joined_uint64(name + "_id", ids.size());
      state.writer->open();
      state.variables_defined = true;
    }

    state.writer->begin_step(static_cast<std::int64_t>(data.curstep), data.curtime);
    state.writer->put_joined_double(name, ids.size(), values.data());
    state.writer->put_joined_uint64(name + "_id", ids.size(), ids.data());
    state.writer->end_step();
  }
#endif

public:
  // constructor
  TracerDiag(PtrInterface interface) : PicChunkDiagWriter(diag_name, interface)
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
  virtual void operator()(json& config) override
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

        const int   size    = ParticleType::get_particle_size();
        const int64 Np      = nbyte / size;
        const int   ndim    = 2;
        const int64 dims[2] = {Np, ParticleType::Nc};

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
