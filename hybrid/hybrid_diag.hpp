// -*- C++ -*-
#ifndef _HYBRID_DIAG_HPP_
#define _HYBRID_DIAG_HPP_

#include <filesystem>
#include <fstream>

namespace hybrid::diag
{
inline std::ofstream open_output(const std::filesystem::path& path)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  return stream;
}
} // namespace hybrid::diag

#endif
