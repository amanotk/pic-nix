// -*- C++ -*-
#ifndef _TRACER_DIAG_HPP_
#define _TRACER_DIAG_HPP_

#include "async.hpp"
#include "nix/xtensor_particle.hpp"

using nix::ParticlePtr;
using nix::ParticleVec;
using ParticleType = nix::ParticlePtr::element_type;

///
/// @brief Diagnostic for picking up tracer
///
template <typename App, typename Data>
class PickupTracerDiag : public BaseDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::basedir;

  // dummy data packer for pickup tracer
  template <typename BasePacker>
  class PickupTracerPacker : public BasePacker
  {
  private:
    int     species;
    float64 xmin;
    float64 xmax;
    float64 ymin;
    float64 ymax;
    float64 zmin;
    float64 zmax;
    float64 fraction;

  public:
    using BasePacker::pack_tracer;

    PickupTracerPacker(json& config)
    {
      const float64 minval = -std::numeric_limits<float64>::max();
      const float64 maxval = +std::numeric_limits<float64>::max();

      species  = config.value("species", 0);
      xmin     = config.value("xmin", minval);
      xmax     = config.value("xmax", maxval);
      ymin     = config.value("ymin", minval);
      ymax     = config.value("ymax", maxval);
      zmin     = config.value("zmin", minval);
      zmax     = config.value("zmax", maxval);
      fraction = config.value("fraction", 0.0);
    }

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
    {
      std::mt19937_64   mt;
      nix::rand_uniform uniform(0.0, 1.0);
      int               Np = data.up[species]->Np;
      auto&             xu = data.up[species]->xu;

      for (int ip = 0; ip < Np; ip++) {
        bool x = xmin <= xu(ip, 0) && xu(ip, 0) <= xmax;
        bool y = ymin <= xu(ip, 1) && xu(ip, 1) <= ymax;
        bool z = zmin <= xu(ip, 2) && xu(ip, 2) <= zmax;
        bool r = uniform(mt) < fraction;

        if (x && y && z && r) {
          int64& id64 = *reinterpret_cast<int64*>(&xu(ip, 6));
          id64        = -std::abs(id64);
        }
      }

      return 0;
    }
  };

public:
  // constructor
  PickupTracerDiag(std::string basedir) : BaseDiag<App, Data>(basedir, "pickup_tracer")
  {
  }

  // data packing functor
  void operator()(json& config, App& app, Data& data) override
  {
    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    auto packer = PickupTracerPacker<XtensorPacker3D>(config);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      data.chunkvec[i]->pack_diagnostic(packer, nullptr, 0);
    }
  }
};

///
/// @brief Diagnostic for tracer
///
template <typename App, typename Data>
class TracerDiag : public AsyncDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::basedir;
  constexpr static int max_species = 10;

  // data packer for particle
  template <typename BasePacker>
  class TracerPacker : public BasePacker
  {
  private:
    int species;
    int seed;

  public:
    using BasePacker::pack_tracer;

    TracerPacker(int species, int seed = 0) : species(species), seed(seed)
    {
    }

    template <typename ChunkData>
    size_t operator()(ChunkData data, uint8_t* buffer, int address)
    {
      return pack_tracer(data.up[species], buffer, address);
    }
  };

public:
  // constructor
  TracerDiag(std::string basedir) : AsyncDiag<App, Data>(basedir, "tracer", max_species)
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

    assert(Ns <= max_species);

    size_t      disp    = 0;
    json        dataset = {};
    std::string fn_data = this->format_filename(config, ".data", "tracer", data.curstep);
    std::string fn_json = this->format_filename(config, ".json", "tracer", data.curstep);
    std::string fn_data_with_path =
        this->format_filename(config, ".data", basedir, ".", "tracer", data.curstep);
    std::string fn_json_with_path =
        this->format_filename(config, ".json", basedir, ".", "tracer", data.curstep);

    if (data.thisrank == 0) {
      this->make_sure_directory_exists(fn_data_with_path);
    }

    this->open_file(fn_data_with_path, &disp, "w");

    //
    // for each particle
    //
    for (int is = 0; is < Ns; is++) {
      // write particles
      size_t disp0 = disp;
      int    seed  = data.thisrank;
      this->launch(is, TracerPacker<XtensorPacker3D>(is, seed), data, disp);

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
        std::ofstream ofs(fn_json_with_path);
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
