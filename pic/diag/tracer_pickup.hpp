// -*- C++ -*-
#ifndef _TRACER_PICKUP_DIAG_HPP_
#define _TRACER_PICKUP_DIAG_HPP_

#include "nix/random.hpp"

#include "chunk_writer.hpp"

///
/// @brief Diagnostic for picking up tracer particles
///
class TracerPickupDiag : public PicDiag
{
public:
  static constexpr const char* diag_name = "tracer_pickup";

protected:
  // dummy data packer for tracer pickup
  class TracerPickupPacker : public PicPacker
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
    TracerPickupPacker(json& config)
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
  TracerPickupDiag(PtrInterface interface) : PicDiag(diag_name, interface)
  {
  }

  // data packing functor
  virtual void operator()(json& config) override
  {
    auto data = interface->get_data();

    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    auto packer = TracerPickupPacker(config);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<PicChunk*>(data.chunkvec[i].get());
      packer(chunk->get_internal_data(), nullptr, 0);
    }
  }
};

#endif
