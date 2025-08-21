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

  /// pack raw array
  template <typename Array, typename Data>
  size_t pack_array_raw(Array& x, Data data, uint8_t* buffer, int address)
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
  template <typename Array, typename Data>
  size_t pack_field(Array& x, Data data, int decimate, uint8_t* buffer, int address)
  {
    auto [size_z, size_y, size_x, size_c] = calc_decimated_size(x, data, decimate);

    size_t count = sizeof(float64) * (size_z * size_y * size_x * size_c) + address;

    if (buffer == nullptr) {
      return count;
    }

    // colocate field components and decimate
    auto y = colocate_field(x, data);
    auto v = decimate_field(y, data, size_z, size_y, size_x, size_c);

    // packing
    float64* ptr = reinterpret_cast<float64*>(buffer + address);
    std::copy(v.begin(), v.end(), ptr);

    return count;
  }

  template <typename Array, typename Data>
  size_t pack_moment(Array& x, Data data, int decimate, uint8_t* buffer, int address)
  {
    auto [size_z, size_y, size_x, size_c] = calc_decimated_size(x, data, decimate);

    size_t count = sizeof(float64) * (size_z * size_y * size_x * size_c) + address;

    if (buffer == nullptr) {
      return count;
    }

    // decimate
    auto v = decimate_field(x, data, size_z, size_y, size_x, size_c);

    // packing
    float64* ptr = reinterpret_cast<float64*>(buffer + address);
    std::copy(v.begin(), v.end(), ptr);

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

  /// calculate decimated size for each dimension
  template <typename Array, typename Data>
  static inline auto calc_decimated_size(Array& x, Data data, int decimate)
  {
    size_t size_z = decimate_size(data.Lbz, data.Ubz, decimate);
    size_t size_y = decimate_size(data.Lby, data.Uby, decimate);
    size_t size_x = decimate_size(data.Lbx, data.Ubx, decimate);

    size_t size_c = 1;
    for (int i = 3; i < x.dimension(); i++) {
      size_c *= x.shape(i);
    }

    return std::make_tuple(size_z, size_y, size_x, size_c);
  }

  /// calculate colocated electromagnetic field components
  template <typename Array, typename Data>
  static inline auto colocate_field(Array& x, Data data)
  {
    bool is_1d = data.Lbz == data.Ubz && data.Lby == data.Uby && data.Lbx != data.Ubx;
    bool is_2d = data.Lbz == data.Ubz && data.Lby != data.Uby && data.Lbx != data.Ubx;
    bool is_3d = data.Lbz != data.Ubz && data.Lby != data.Uby && data.Lbx != data.Ubx;

    if (is_1d) {
      return colocate_field_1d(x, data);
    }

    if (is_2d) {
      return colocate_field_2d(x, data);
    }

    if (is_3d) {
      return colocate_field_3d(x, data);
    }

    // must be unreachable
    assert(false);
  }

  /// decimate field
  template <typename Array, typename Data>
  static inline auto decimate_field(Array& x, Data data, size_t size_z, size_t size_y,
                                    size_t size_x, size_t size_c)
  {
    // block size
    int     blocksize_z = (data.Ubz - data.Lbz + 1) / size_z;
    int     blocksize_y = (data.Uby - data.Lby + 1) / size_y;
    int     blocksize_x = (data.Ubx - data.Lbx + 1) / size_x;
    float64 factor      = 1.0 / (blocksize_z * blocksize_y * blocksize_x);

    // prepare reshaped input and output arrays
    auto y_shape = std::array<std::size_t, 4>{x.shape(0), x.shape(1), x.shape(2), size_c};
    auto z_shape = std::array<std::size_t, 4>{size_z, size_y, size_x, size_c};
    auto y       = xt::reshape_view(x, y_shape);
    auto z       = xt::eval(xt::zeros<float64>(z_shape));

    // loop for each grid in block
    for (int iz_block = 0; iz_block < blocksize_z; iz_block++) {
      for (int iy_block = 0; iy_block < blocksize_y; iy_block++) {
        for (int ix_block = 0; ix_block < blocksize_x; ix_block++) {
          // loop for each block
          for (int jz = 0; jz < size_z; jz++) {
            for (int jy = 0; jy < size_y; jy++) {
              for (int jx = 0; jx < size_x; jx++) {
                int iz = data.Lbz + jz * blocksize_z + iz_block;
                int iy = data.Lby + jy * blocksize_y + iy_block;
                int ix = data.Lbx + jx * blocksize_x + ix_block;
                for (size_t ic = 0; ic < size_c; ic++) {
                  z(jz, jy, jx, ic) += factor * y(iz, iy, ix, ic);
                }
              }
            }
          }
        }
      }
    }

    return z;
  }

  static inline size_t decimate_size(int Lb, int Ub, int decimate)
  {
    int size = static_cast<size_t>(Ub - Lb + 1);

    if (size <= decimate) {
      return 1;
    } else {
      return size / decimate;
    }
  }

  template <typename Array, typename Data>
  static inline auto colocate_field_1d(Array& x, Data data)
  {
    auto y = xt::zeros_like(x);

    int iz = data.Lbz;
    int iy = data.Lby;

    for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
      // Ex, Ey, Ez
      y(iz, iy, ix, 0) = 0.50 * (x(iz, iy, ix, 0) + x(iz, iy, ix + 1, 0));
      y(iz, iy, ix, 1) = x(iz, iy, ix, 1);
      y(iz, iy, ix, 2) = x(iz, iy, ix, 2);
      // Bx, By, Bz
      y(iz, iy, ix, 3) = x(iz, iy, ix, 3);
      y(iz, iy, ix, 4) = 0.50 * (x(iz, iy, ix, 4) + x(iz, iy, ix + 1, 4));
      y(iz, iy, ix, 5) = 0.50 * (x(iz, iy, ix, 5) + x(iz, iy, ix + 1, 5));
    }

    return y;
  }

  template <typename Array, typename Data>
  static inline auto colocate_field_2d(Array& x, Data data)
  {
    auto y = xt::zeros_like(x);

    int iz = data.Lbz;

    for (int iy = data.Lby; iy <= data.Uby; iy++) {
      for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
        // Ex, Ey, Ez
        y(iz, iy, ix, 0) = 0.50 * (x(iz, iy, ix, 0) + x(iz, iy, ix + 1, 0));
        y(iz, iy, ix, 1) = 0.50 * (x(iz, iy, ix, 1) + x(iz, iy + 1, ix, 1));
        y(iz, iy, ix, 2) = x(iz, iy, ix, 2);
        // Bx, By, Bz
        y(iz, iy, ix, 3) = 0.50 * (x(iz, iy, ix, 3) + x(iz, iy + 1, ix, 3));
        y(iz, iy, ix, 4) = 0.50 * (x(iz, iy, ix, 4) + x(iz, iy, ix + 1, 4));
        y(iz, iy, ix, 5) = 0.25 * (x(iz, iy, ix, 5) + x(iz, iy + 1, ix + 1, 5) +
                                   x(iz, iy, ix + 1, 5) + x(iz, iy + 1, ix, 5));
      }
    }

    return y;
  }

  template <typename Array, typename Data>
  static inline auto colocate_field_3d(Array& x, Data data)
  {
    auto y = xt::zeros_like(x);

    for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
      for (int iy = data.Lby; iy <= data.Uby; iy++) {
        for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
          // Ex, Ey, Ez
          y(iz, iy, ix, 0) = 0.50 * (x(iz, iy, ix, 0) + x(iz, iy, ix + 1, 0));
          y(iz, iy, ix, 1) = 0.50 * (x(iz, iy, ix, 1) + x(iz, iy + 1, ix, 1));
          y(iz, iy, ix, 2) = 0.50 * (x(iz, iy, ix, 2) + x(iz + 1, iy, ix, 2));
          // Bx, By, Bz
          y(iz, iy, ix, 3) = 0.25 * (x(iz, iy, ix, 3) + x(iz + 1, iy + 1, ix, 3) +
                                     x(iz, iy + 1, ix, 3) + x(iz + 1, iy, ix, 3));
          y(iz, iy, ix, 4) = 0.25 * (x(iz, iy, ix, 4) + x(iz + 1, iy, ix + 1, 4) +
                                     x(iz + 1, iy, ix, 4) + x(iz, iy, ix + 1, 4));
          y(iz, iy, ix, 5) = 0.25 * (x(iz, iy, ix, 5) + x(iz, iy + 1, ix + 1, 5) +
                                     x(iz, iy, ix + 1, 5) + x(iz, iy + 1, ix, 5));
        }
      }
    }

    return y;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
