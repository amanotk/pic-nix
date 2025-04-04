// -*- C++ -*-
#ifndef _PARTICLE_DIAG_HPP_
#define _PARTICLE_DIAG_HPP_

#include "parallel.hpp"

///
/// @brief Diagnostic for particle
///
class ParticleDiag : public ParallelDiag
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

public:
  // constructor
  ParticleDiag(app_type& application) : ParallelDiag(diag_name, application)
  {
  }

  // data packing functor
  void operator()(json& config) override
  {
    auto data = application.get_internal_data();
    auto Ns   = application.get_num_species();

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
      size_t  nbyte    = this->queue(packer, data, disp);

      // meta data
      {
        std::string name = tfm::format("up%02d", is);
        std::string desc = tfm::format("particle species %02d", is);

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
                      {"order", 1},
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
