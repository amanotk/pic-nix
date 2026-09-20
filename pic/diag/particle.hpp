// -*- C++ -*-
#ifndef _PARTICLE_DIAG_HPP_
#define _PARTICLE_DIAG_HPP_

#include "chunk_writer.hpp"
#include "nix/diag/adios.hpp"

#include <map>

///
/// @brief Diagnostic for particle
///
class ParticleDiag : public PicChunkDiagWriter
{
public:
  static constexpr const char* diag_name = "particle";

protected:
  // data packer for particle
  class ParticlePacker : public PicPacker
  {
  private:
    int     species;
    int     seed;
    float64 fraction;

  public:
    ParticlePacker(int species, int seed = 0, float64 fraction = 1.0)
        : species(species), seed(seed), fraction(fraction)
    {
    }

    std::vector<int64_t> generate_random_index(int N, int M, int random_seed)
    {
      std::mt19937_64 engine(random_seed);

      std::vector<int64_t> index(N);
      std::iota(index.begin(), index.end(), 0);

      if (M < N) {
        // randomly pick M out of N particles
        std::shuffle(index.begin(), index.end(), engine);
        index.resize(M);
        std::sort(index.begin(), index.end());
      }

      return index;
    }

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      const int N     = data.up[species]->Np;
      const int M     = std::min(static_cast<int>(N * fraction), N);
      auto      index = generate_random_index(N, M, seed);
      return pack_particle(data.up[species], index, buffer, address);
    }
  };

  struct AdiosState {
    std::unique_ptr<nix::AdiosWriter> writer;
    bool                              variables_defined = false;
  };

  std::map<std::string, AdiosState> adios_state;

  void write_adios(json& config)
  {
    auto data = interface->get_data();
    if (this->require_diagnostic(data.curstep, config) == false) {
      return;
    }

    const std::string                       prefix   = this->get_prefix(config, "particle");
    const float64                           fraction = config.value("fraction", 0.01);
    const int                               Ns       = interface->get_num_species();
    const size_t                            width    = ParticleType::Nc - 1;
    auto&                                   state    = adios_state[prefix];
    std::vector<std::vector<float64>>       values(static_cast<size_t>(Ns));
    std::vector<std::vector<std::uint64_t>> ids(static_cast<size_t>(Ns));

    for (int is = 0; is < Ns; is++) {
      ParticlePacker packer(is, data.thisrank, fraction);
      for (int i = 0; i < data.chunkvec.size(); i++) {
        auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
        auto cdata = chunk->get_internal_data();
        auto index = packer.generate_random_index(
            cdata.up[is]->get_Np_active(),
            std::min(static_cast<int>(cdata.up[is]->get_Np_active() * fraction),
                     cdata.up[is]->get_Np_active()),
            data.thisrank);

        values[is].reserve(values[is].size() + index.size() * width);
        ids[is].reserve(ids[is].size() + index.size());
        for (const auto particle_index : index) {
          for (size_t ic = 0; ic < width; ic++) {
            values[is].push_back(cdata.up[is]->xu(particle_index, ic));
          }
          std::uint64_t id = 0;
          std::memcpy(&id, &cdata.up[is]->xu(particle_index, ParticleType::Nc - 1), sizeof(id));
          ids[is].push_back(id);
        }
      }
    }

    if (state.writer == nullptr) {
      state.writer = std::make_unique<nix::AdiosWriter>(this->info);
      state.writer->initialize("particle", prefix, interface->get_configuration());
      for (int is = 0; is < Ns; is++) {
        const std::string name = fmt::format("up{:02d}", is);
        state.writer->define_joined_double(name, width, ids[is].size());
        state.writer->define_joined_uint64(name + "_id", ids[is].size());
      }
      state.writer->open();
      state.variables_defined = true;
    }

    if (state.variables_defined == false) {
      throw std::logic_error("ADIOS2 particle variables were not initialized");
    }

    state.writer->begin_step(static_cast<std::int64_t>(data.curstep), data.curtime);
    for (int is = 0; is < Ns; is++) {
      const std::string name = fmt::format("up{:02d}", is);
      state.writer->put_joined_double(name, ids[is].size(), values[is].data());
      state.writer->put_joined_uint64(name + "_id", ids[is].size(), ids[is].data());
    }
    state.writer->end_step();
  }

public:
  // constructor
  ParticleDiag(PtrInterface interface) : PicChunkDiagWriter(diag_name, interface)
  {
  }

  void shutdown() override
  {
    for (auto& [prefix, state] : adios_state) {
      if (state.writer != nullptr) {
        state.writer->close();
      }
    }
  }

protected:
  void write_file(json& config)
  {
    auto data = interface->get_data();
    auto Ns   = interface->get_num_species();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    size_t      disp    = 0;
    json        dataset = {};
    std::string prefix  = this->get_prefix(config, "particle");
    std::string dirname = this->format_dirname(prefix);
    std::string fn_data = this->format_filename("", ".data", data.curstep);
    std::string fn_json = this->format_filename("", ".json", data.curstep);

    this->make_sure_directory_exists(dirname + fn_data);
    this->open_file(dirname + fn_data, &disp, "w");

    //
    // for each particle
    //
    for (int is = 0; is < Ns; is++) {
      // write particles
      int     seed     = data.thisrank;
      float64 fraction = config.value("fraction", 0.01);
      auto    packer   = ParticlePacker(is, seed, fraction);
      size_t  disp0    = disp;
      size_t  nbyte    = this->write_packed_chunks(packer, data, disp);

      // meta data
      {
        std::string name = fmt::format("up{:02d}", is);
        std::string desc = fmt::format("particle species {:02d}", is);

        const int   size    = ParticleType::get_particle_size();
        const int64 Np      = nbyte / size;
        const int   ndim    = 2;
        const int64 dims[2] = {Np, ParticleType::Nc};

        nixio::put_metadata(dataset, name, "f8", desc, disp0, nbyte, ndim, dims);
      }
    }

    this->close_file();

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

public:
  // data packing functor
  void operator()(json& config) override
  {
    if (this->info->iomode == "adios") {
      write_adios(config);
    } else {
      write_file(config);
    }
  }
};

#endif
