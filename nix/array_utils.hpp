// -*- C++ -*-
#ifndef _ARRAY_UTILS_HPP_
#define _ARRAY_UTILS_HPP_

#include "nix.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>

NIX_NAMESPACE_BEGIN

// clang-format off
template <typename T, typename = void>
struct is_mdspan : std::false_type {};

// determine if T is a mdspan type
template <typename T>
struct is_mdspan<T, std::void_t
  <
    decltype(std::declval<T>().data_handle()),   // data pointer
    decltype(std::declval<T>().size()),          // total size
    decltype(std::declval<T>().stride(0)),       // stride of dimension 0
    decltype(std::declval<T>().extents().rank()) // number of dimensions
  >> : std::true_type {
};

template <typename T, typename = void>
struct is_xtensor : std::false_type {};

// determine if T is a xtensor type
template <typename T>
struct is_xtensor<T, std::void_t
  <
    decltype(std::declval<T>().data()),       // data pointer
    decltype(std::declval<T>().size()),       // total size
    decltype(std::declval<T>().strides()[0]), // stride of dimension 0
    decltype(std::declval<T>().dimension())   // number of dimensions
  >> : std::true_type {
};
// clang-format on

/// utility to get internal data pointer of array
template <typename T_array>
static auto get_data_pointer(T_array& array)
{
  if constexpr (is_xtensor<T_array>::value == true) {
    return array.data();
  } else if constexpr (is_mdspan<T_array>::value == true) {
    return array.data_handle();
  }
}

/// utility to get extent of array for a given dimension
template <typename T_array>
static auto get_extent(T_array& array, int dim)
{
  if constexpr (is_xtensor<T_array>::value == true) {
    return array.shape(dim);
  } else if constexpr (is_mdspan<T_array>::value == true) {
    return array.extent(dim);
  }
}

/// utility to get stride of array for a given dimension
template <typename T_array>
static auto get_stride(T_array& array, int dim)
{
  if constexpr (is_xtensor<T_array>::value == true) {
    return array.strides()[dim];
  } else if constexpr (is_mdspan<T_array>::value == true) {
    return array.stride(dim);
  }
}

/// fill array with given value
template <typename T_array, typename T>
void fill_all(T_array& array, T&& value)
{
  std::fill_n(get_data_pointer(array), array.size(), value);
}

NIX_NAMESPACE_END

#endif
