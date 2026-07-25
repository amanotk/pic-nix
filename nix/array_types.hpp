// -*- C++ -*-
#ifndef _ARRAY_TYPES_HPP_
#define _ARRAY_TYPES_HPP_

#include "nix.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <array>
#include <cstddef>

NIX_NAMESPACE_BEGIN

//
// dynamic-shape array aliases (backend-switchable)
//

template <typename T>
using Array1D = xt::xtensor<T, 1>;

template <typename T>
using Array2D = xt::xtensor<T, 2>;

template <typename T>
using Array3D = xt::xtensor<T, 3>;

template <typename T>
using Array4D = xt::xtensor<T, 4>;

template <typename T>
using Array5D = xt::xtensor<T, 5>;

//
// fixed-shape arrays (std::array-based, not backend-switchable)
//

template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
struct FixedArray3D {
  static constexpr std::size_t size_ = D0 * D1 * D2;

  std::array<T, size_> data_{};

  T& operator()(int z, int y, int x)
  {
    return data_[static_cast<std::size_t>(z) * D1 * D2 + static_cast<std::size_t>(y) * D2 +
                 static_cast<std::size_t>(x)];
  }
  const T& operator()(int z, int y, int x) const
  {
    return data_[static_cast<std::size_t>(z) * D1 * D2 + static_cast<std::size_t>(y) * D2 +
                 static_cast<std::size_t>(x)];
  }
  T* data()
  {
    return data_.data();
  }
  const T* data() const
  {
    return data_.data();
  }
  constexpr std::size_t size() const
  {
    return size_;
  }
  void fill(const T& v)
  {
    data_.fill(v);
  }
};

template <typename T, std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3>
struct FixedArray4D {
  static constexpr std::size_t size_ = D0 * D1 * D2 * D3;

  std::array<T, size_> data_{};

  T& operator()(int w, int z, int y, int x)
  {
    return data_[static_cast<std::size_t>(w) * D1 * D2 * D3 +
                 static_cast<std::size_t>(z) * D2 * D3 + static_cast<std::size_t>(y) * D3 +
                 static_cast<std::size_t>(x)];
  }
  const T& operator()(int w, int z, int y, int x) const
  {
    return data_[static_cast<std::size_t>(w) * D1 * D2 * D3 +
                 static_cast<std::size_t>(z) * D2 * D3 + static_cast<std::size_t>(y) * D3 +
                 static_cast<std::size_t>(x)];
  }
  T* data()
  {
    return data_.data();
  }
  const T* data() const
  {
    return data_.data();
  }
  constexpr std::size_t size() const
  {
    return size_;
  }
  void fill(const T& v)
  {
    data_.fill(v);
  }
};

NIX_NAMESPACE_END

#endif
