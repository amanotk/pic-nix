// -*- C++ -*-
#ifndef _NIX_HPP_
#define _NIX_HPP_

#include "debug.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <sys/time.h>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <toml.hpp>
#include <xsimd/xsimd.hpp>

#ifdef _POSIX_VERSION
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAS_MPI_THREAD_MULTIPLE
#define OMP_MAYBE_CRITICAL
constexpr int NIX_MPI_THREAD_LEVEL = MPI_THREAD_MULTIPLE;
#else
#define OMP_MAYBE_CRITICAL _Pragma("omp critical")
constexpr int NIX_MPI_THREAD_LEVEL = MPI_THREAD_SERIALIZED;
#endif

#define NIX_NAMESPACE_BEGIN                                                                        \
  namespace nix                                                                                    \
  {
#define NIX_NAMESPACE_END }

//
// nix namespace
//
NIX_NAMESPACE_BEGIN

// json
using json = nlohmann::ordered_json;

//
// typedefs namespace
//
namespace typedefs
{
// integer types
using int32  = int32_t;
using int64  = int64_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// floating point number types
using float32 = float;
using float64 = double;
using real    = float64;

// SIMD types
using simd_f32 = xsimd::batch<float32>;
using simd_f64 = xsimd::batch<float64>;
using simd_i32 = xsimd::batch<int32>;
using simd_i64 = xsimd::batch<int64>;

// MPI datatype for consistent notations
// (constexpr does not work with some MPI library)
const MPI_Datatype MPI_FLOAT32_T = MPI_FLOAT;
const MPI_Datatype MPI_FLOAT64_T = MPI_DOUBLE;
} // namespace typedefs

using namespace typedefs;

// maximum number of chunk per MPI rank
// (used to make MPI send/recv tags less than this value)
constexpr int MAX_CHUNK_PER_RANK = 32768;

//
// SIMD width (512 bit for 64bit float by default)
//
#ifndef NIX_SIMD_WIDTH
#define NIX_SIMD_WIDTH 8
#endif
constexpr int nix_simd_width = NIX_SIMD_WIDTH;

// mathematical constants
namespace math
{
constexpr float64 pi  = M_PI;     ///< pi
constexpr float64 pi2 = 2 * M_PI; ///< 2 pi
constexpr float64 pi4 = 4 * M_PI; ///< 4 pi
} // namespace math

// binary mode
const std::ios::openmode binary_write  = std::ios::binary | std::ios::out | std::ios::trunc;
const std::ios::openmode binary_append = std::ios::binary | std::ios::out | std::ios::app;
const std::ios::openmode binary_read   = std::ios::binary | std::ios::in;

// text mode
const std::ios::openmode text_write  = std::ios::out | std::ios::trunc;
const std::ios::openmode text_append = std::ios::out | std::ios::app;
const std::ios::openmode text_read   = std::ios::in;

// send/recv mode
enum SendRecvMode {
  SendMode = 0b01000000000000, // 4096
  RecvMode = 0b10000000000000, // 8192
};

///
/// @brief return wall clock time since epoch
/// @return time in second
///
inline double wall_clock()
{
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch());
  return static_cast<double>(t.count()) * 1.0e-9;
}

///
/// @brief return flag to determine endian of the system
/// @return 1 on little endian and 16777216 on big endian
///
inline int32_t get_endian_flag()
{
  union {
    int32_t flag;
    uint8_t byte[4] = {1, 0, 0, 0};
  } endian_flag;

  return endian_flag.flag;
}

inline int get_max_threads()
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int sync_directory(std::string dirname)
{
  int status = 0;

#ifdef _POSIX_VERSION
  if (std::filesystem::is_directory(dirname) == true) {
    int fd = open(dirname.c_str(), O_RDONLY | O_DIRECTORY);
    status = fsync(fd);
    close(fd);
  }
#endif

  return status;
}

///
/// @brief convenient wrapper for std::memcpy
/// @param dst destination buffer pointer
/// @param src source buffer pointer
/// @param count number of bytes to be copied
/// @param dstaddr offset from dst pointer
/// @param srcaddr offset from src pointer
/// @return number of bytes copied
///
/// If both `dst` or `src` are not `nullptr`, then data copy will be performed and it return the
/// number of bytes that are copied from `src` to `dst`. Otherwise, it will be in a query mode, in
/// which the copy operation will not be performed but the number of bytes to be copied will be
/// returned.
/// For coninience, `dstaddr` and `srcaddr` can be specified as offsets to the buffer pointers.
///
inline size_t memcpy_count(void* dst, void* src, size_t count, size_t dstaddr, size_t srcaddr)
{
  if (dst != nullptr && src != nullptr) {
    uint8_t* dstptr = &static_cast<uint8_t*>(dst)[dstaddr];
    uint8_t* srcptr = &static_cast<uint8_t*>(src)[srcaddr];
    std::memcpy(dstptr, srcptr, count);
  }
  return count;
}

///
/// @brief return string representation of given step
/// @param step time step
/// @return formatted string
///
inline std::string format_step(int step)
{
  return tfm::format("%08d", step);
}

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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
