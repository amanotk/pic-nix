// -*- C++ -*-
#ifndef _TRACER_DIAG_HPP_
#define _TRACER_DIAG_HPP_

#include "nix/random.hpp"

#include "parallel.hpp"

///
/// @brief Diagnostic for picking up tracer
///
class PickupTracerDiag : public PicDiag
{
public:
  static constexpr const char* diag_name = "pickup_tracer";

protected:
  // dummy data packer for pickup tracer
  class PickupTracerPacker : public PicPacker
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

    virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address) override
    {
      std::random_device rd;
      std::mt19937_64    mt(rd());
      nix::rand_uniform  uniform(0.0, 1.0);
      int                Np = data.up[species]->Np;
      auto&              xu = data.up[species]->xu;

      for (int ip = 0; ip < Np; ip++) {
        bool x = xmin <= xu(ip, 0) && xu(ip, 0) <= xmax;
        bool y = ymin <= xu(ip, 1) && xu(ip, 1) <= ymax;
        bool z = zmin <= xu(ip, 2) && xu(ip, 2) <= zmax;
        bool r = uniform(mt) < fraction;

        if (x && y && z && r) {
          // make ID negative
          int64 id64;
          std::memcpy(&id64, &xu(ip, 6), sizeof(int64));
          id64 = -std::abs(id64);
          std::memcpy(&xu(ip, 6), &id64, sizeof(int64));
        }
      }

      return 0;
    }
  };

public:
  // constructor
  PickupTracerDiag(app_type& application) : PicDiag(diag_name, application)
  {
  }

  // data packing functor
  virtual void operator()(json& config) override
  {
    auto data = application.get_internal_data();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    auto packer = PickupTracerPacker(config);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto chunk_data = data.chunkvec[i]->get_internal_data();
      packer(chunk_data, nullptr, 0);
    }
  }
};

///
/// @brief Diagnostic for tracer
///
class TracerDiag : public ParallelDiag
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
  TracerDiag(app_type& application) : ParallelDiag(diag_name, application)
  {
  }

  // data packing functor
  virtual void operator()(json& config) override
  {
    auto data = application.get_internal_data();

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
      size_t nbyte   = this->queue(packer, data, disp);

      // meta data
      {
        std::string name = tfm::format("up%02d", species);
        std::string desc = tfm::format("tracer particle species %02d", species);

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
