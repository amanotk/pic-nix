// -*- C++ -*-
#ifndef _XTENSOR_PACKER3D_HPP_
#define _XTENSOR_PACKER3D_HPP_

#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Data Packer3D for data output
///
class XtensorPacker3D
{
public:
  /// pack coordinate
  template <typename Array>
  size_t pack_coordinate(int& Lb, int& Ub, Array& x, uint8_t* buffer, int address)
  {
    size_t size  = Ub - Lb + 1;
    size_t count = sizeof(float64) * size + address;

    if (buffer == nullptr) {
      return count;
    }

    // packing
    float64* ptr   = reinterpret_cast<float64*>(buffer + address);
    auto     coord = xt::view(x, xt::range(Lb, Ub + 1));
    std::copy(coord.begin(), coord.end(), ptr);

    return count;
  }

  /// pack field
  template <typename Array, typename Data>
  size_t pack_field(Array& x, Data data, uint8_t* buffer, int address)
  {
    // calculate number of elements
    size_t size = (data.Ubz - data.Lbz + 1) * (data.Uby - data.Lby + 1) * (data.Ubx - data.Lbx + 1);
    for (int i = 3; i < x.dimension(); i++) {
      size *= x.shape(i);
    }

    size_t count = sizeof(float64) * size + address;

    if (buffer == nullptr) {
      return count;
    }

    auto Iz = xt::range(data.Lbz, data.Ubz + 1);
    auto Iy = xt::range(data.Lby, data.Uby + 1);
    auto Ix = xt::range(data.Lbx, data.Ubx + 1);
    auto vv = xt::strided_view(x, {Iz, Iy, Ix, xt::ellipsis()});

    // packing
    float64* ptr = reinterpret_cast<float64*>(buffer + address);
    std::copy(vv.begin(), vv.end(), ptr);

    return count;
  }

  /// pack particle
  template <typename ParticlePtr, typename Array>
  size_t pack_particle(ParticlePtr& p, Array& index, uint8_t* buffer, int address)
  {
    size_t count = address;

    auto&  xu   = p->xu;
    size_t size = ParticlePtr::element_type::get_particle_size();
    for (int i = 0; i < index.size(); i++) {
      count += memcpy_count(buffer, &xu(index[i], 0), size, count, 0);
    }

    return count;
  }

  /// pack tracer
  template <typename ParticlePtr>
  size_t pack_tracer(ParticlePtr& p, uint8_t* buffer, int address)
  {
    size_t count = address;

    auto&  xu   = p->xu;
    size_t np   = p->get_Np_active();
    size_t size = ParticlePtr::element_type::get_particle_size();
    for (int i = 0; i < np; i++) {
      int64 id64;
      std::memcpy(&id64, &xu(i, 6), sizeof(int64));
      if (id64 < 0) {
        count += memcpy_count(buffer, &xu(i, 0), size, count, 0);
      }
    }

    return count;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
