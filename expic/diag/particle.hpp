// -*- C++ -*-
#ifndef _PARTICLE_DIAG_HPP_
#define _PARTICLE_DIAG_HPP_

#include "async.hpp"
#include "nix/xtensor_particle.hpp"

using nix::ParticlePtr;
using nix::ParticleVec;
using ParticleType = nix::ParticlePtr::element_type;

///
/// @brief Diagnostic for particle
///
template <typename App, typename Data>
class ParticleDiag : public AsyncDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::info;

  // data packer for particle
  template <typename BasePacker>
  class ParticlePacker : public BasePacker
  {
  private:
    int     species;
    int     seed;
    float64 fraction;

  public:
    using BasePacker::pack_particle;

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

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
    {
      const int N     = data.up[species]->Np;
      const int M     = std::min(static_cast<int>(N * fraction), N);
      auto      index = generate_random_index(N, M, seed);
      return pack_particle(data.up[species], index, buffer, address);
    }
  };

public:
  // constructor
  ParticleDiag(std::shared_ptr<DiagInfo> info) : AsyncDiag<App, Data>("particle", info)
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
    const int nc = data.cdims[3];
    const int Ns = app.get_Ns();

    this->set_queue_size(Ns);

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
      size_t  disp0    = disp;
      int     seed     = data.thisrank;
      float64 fraction = config.value("fraction", 0.01);
      this->launch(is, ParticlePacker<XtensorPacker3D>(is, seed, fraction), data, disp);

      // meta data
      {
        std::string name = tfm::format("up%02d", is);
        std::string desc = tfm::format("particle species %02d", is);

        const int size    = ParticleType::get_particle_size();
        const int Np      = (disp - disp0) / size;
        const int ndim    = 2;
        const int dims[2] = {Np, ParticleType::Nc};

        nixio::put_metadata(dataset, name, "f8", desc, disp0, Np * size, ndim, dims);
      }
    }

    if (this->is_completed() == true) {
      this->close_file();
    }

    //
    // output json file
    //
    {
      json root;

      // meta data
      root["meta"] = {{"endian", nix::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"order", 1},
                      {"time", data.curtime},
                      {"step", data.curstep}};
      // dataset
      root["dataset"] = dataset;

      if (data.thisrank == 0) {
        std::ofstream ofs(dirname + fn_json);
        ofs << std::setw(2) << root;
        ofs.close();
      }

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
