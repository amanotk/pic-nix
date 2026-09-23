// -*- C++ -*-
#ifndef _PIC_ASCENT_BLUEPRINT_BUILDER_HPP_
#define _PIC_ASCENT_BLUEPRINT_BUILDER_HPP_

#include "domain_view.hpp"

#include <conduit.hpp>

#include <deque>
#include <vector>

namespace pic_ascent
{
struct BlueprintOptions {
  bool electric_field  = true;
  bool magnetic_field  = true;
  bool mass_current    = true;
  bool energy_momentum = false;
  bool raw_fields      = false;
  bool raw_particles   = false;
};

struct BlueprintPublication {
  conduit::Node                    node;
  std::deque<std::vector<float64>> buffers;
};

class BlueprintBuilder
{
public:
  static BlueprintPublication build(const std::vector<PicChunk*>& chunks, int cycle, float64 time,
                                    const json&             configuration,
                                    const BlueprintOptions& options = {});
};
} // namespace pic_ascent

#endif
