// -*- C++ -*-
#ifndef _DIAG_METADATA_HPP_
#define _DIAG_METADATA_HPP_

#include "nix.hpp"
#include "nixio.hpp"

NIX_NAMESPACE_BEGIN

/// make common diagnostic metadata
inline json make_metadata(std::string rawfile, float64 time, int step)
{
  json meta       = {};
  meta["endian"]  = nix::get_endian_flag();
  meta["rawfile"] = rawfile;
  meta["layout"]  = nix::ARRAY_LAYOUT;
  meta["time"]    = time;
  meta["step"]    = step;
  return meta;
}

/// make common diagnostic metadata with chunk id range
inline json make_metadata(std::string rawfile, float64 time, int step,
                          const std::vector<int>& chunk_id_range)
{
  json meta              = make_metadata(rawfile, time, step);
  meta["chunk_id_range"] = chunk_id_range;
  return meta;
}

NIX_NAMESPACE_END

#endif
