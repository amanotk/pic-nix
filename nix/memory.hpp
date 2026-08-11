// -*- C++ -*-
#ifndef _NIX_MEMORY_HPP_
#define _NIX_MEMORY_HPP_

#include "nix.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__linux__) || defined(_POSIX_VERSION)
#include <unistd.h>
#endif

NIX_NAMESPACE_BEGIN

#ifndef NIX_MMAP_THRESHOLD
#define NIX_MMAP_THRESHOLD (1024 * 1024)
#endif

///
/// @brief return the live resident set size of this process in bytes
/// @return resident set size in bytes; 0 when it cannot be determined
/// @note reads the VmRSS entry of /proc/self/status, which is
/// Linux-specific; the value is 0 on other platforms. This is the
/// per-process live RSS, suitable for per-rank accounting; getrusage's
/// ru_maxrss reports the peak, not the live value.
///
inline int64_t get_process_rss()
{
  int64_t resident = 0;

#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string   key;
  long long     value_kb = 0;

  while (status >> key) {
    if (key == "VmRSS:") {
      if (status >> value_kb) {
        resident = value_kb * 1024;
      }
      break;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
#endif

  return resident;
}

///
/// @brief configure the default C malloc arena policy
/// @note on glibc, the adaptive mmap threshold otherwise rises after repeated
/// large alloc/free cycles (e.g. particle buffer resizes and chunk
/// rebalancing), keeping freed large blocks resident in the heap arena.
/// Setting NIX_MMAP_THRESHOLD makes large freed buffers return to the OS.
/// This is a no-op on other libcs and is inert while an LD_PRELOAD
/// allocator (mimalloc, jemalloc, tcmalloc) replaces the C malloc.
///
inline void configure_allocator()
{
#if defined(__GLIBC__)
  mallopt(M_MMAP_THRESHOLD, NIX_MMAP_THRESHOLD);
#endif
}

NIX_NAMESPACE_END

#endif
