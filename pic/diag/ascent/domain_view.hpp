// -*- C++ -*-
#ifndef _PIC_ASCENT_DOMAIN_VIEW_HPP_
#define _PIC_ASCENT_DOMAIN_VIEW_HPP_

#include "pic/pic_chunk.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pic_ascent
{
struct RawArrayView {
  float64*                   data = nullptr;
  std::array<std::size_t, 3> shape_zyx{};
  std::size_t                component_count = 0;
};

struct ParticleView {
  float64*    data      = nullptr;
  std::size_t np_active = 0;
};

struct DomainView {
  PicChunk*                 source    = nullptr;
  int                       domain_id = 0;
  int                       dimension = 0;
  int                       cycle     = 0;
  float64                   time      = 0.0;
  std::array<int, 3>        global_cell_shape_zyx{};
  std::array<int, 3>        local_cell_shape_zyx{};
  std::array<int, 3>        global_offset_zyx{};
  std::array<int, 3>        active_lower_zyx{};
  std::array<int, 3>        active_upper_zyx{};
  std::array<bool, 3>       active_axes_zyx{};
  int                       boundary_margin = 0;
  std::array<float64, 3>    spacing_xyz{};
  std::array<float64, 3>    physical_origin_xyz{};
  RawArrayView              uf;
  RawArrayView              uj;
  std::array<int, 27>       neighbor_domain_ids{};
  std::array<int, 27>       neighbor_ranks{};
  std::vector<ParticleView> particles;

private:
  template <typename Array>
  static RawArrayView make_raw_view(Array& array, const std::array<bool, 3>& active_axes,
                                    int boundary_margin)
  {
    const std::array<std::size_t, 3> source_shape = {
        array.shape(0),
        array.shape(1),
        array.shape(2),
    };
    const std::array<std::size_t, 3> first = {
        active_axes[0] ? 0 : static_cast<std::size_t>(boundary_margin),
        active_axes[1] ? 0 : static_cast<std::size_t>(boundary_margin),
        active_axes[2] ? 0 : static_cast<std::size_t>(boundary_margin),
    };
    const std::size_t component_count = array.shape(3);
    const std::size_t offset =
        ((first[0] * source_shape[1] + first[1]) * source_shape[2] + first[2]) * component_count;

    return {
        array.data() + offset,
        {active_axes[0] ? source_shape[0] : 1, active_axes[1] ? source_shape[1] : 1,
         active_axes[2] ? source_shape[2] : 1},
        component_count,
    };
  }

  static int global_extent(float64 lower, float64 upper, float64 spacing)
  {
    if (spacing == 0.0) {
      return 0;
    }
    return static_cast<int>(std::lround((upper - lower) / spacing));
  }

public:
  DomainView(PicChunk& chunk, int cycle, float64 time)
      : source(&chunk), domain_id(chunk.get_id()), cycle(cycle), time(time)
  {
    auto data   = chunk.get_internal_data();
    auto dims   = chunk.get_dims();
    auto offset = chunk.get_offset();
    auto xrange = chunk.get_xrange_global();
    auto yrange = chunk.get_yrange_global();
    auto zrange = chunk.get_zrange_global();

    active_axes_zyx = {chunk.has_zdim(), chunk.has_ydim(), chunk.has_xdim()};
    dimension       = static_cast<int>(active_axes_zyx[0]) + static_cast<int>(active_axes_zyx[1]) +
                static_cast<int>(active_axes_zyx[2]);
    global_cell_shape_zyx = {global_extent(zrange.first, zrange.second, data.delz),
                             global_extent(yrange.first, yrange.second, data.dely),
                             global_extent(xrange.first, xrange.second, data.delx)};
    local_cell_shape_zyx  = {dims[0], dims[1], dims[2]};
    global_offset_zyx     = {offset[0], offset[1], offset[2]};
    active_lower_zyx      = {data.Lbz, data.Lby, data.Lbx};
    active_upper_zyx      = {data.Ubz, data.Uby, data.Ubx};
    boundary_margin       = data.boundary_margin;
    spacing_xyz           = {data.delx, data.dely, data.delz};
    physical_origin_xyz   = {data.xlim[0], data.ylim[0], data.zlim[0]};

    uf = make_raw_view(data.uf, active_axes_zyx, boundary_margin);
    uj = make_raw_view(data.uj, active_axes_zyx, boundary_margin);

    std::size_t index = 0;
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          neighbor_domain_ids[index] = chunk.get_nb_id(dz, dy, dx);
          neighbor_ranks[index]      = chunk.get_nb_rank(dz, dy, dx);
          index++;
        }
      }
    }

    particles.reserve(data.up.size());
    for (const auto& particle : data.up) {
      ParticleView view;
      if (particle != nullptr) {
        view.data      = particle->xu.data();
        view.np_active = static_cast<std::size_t>(particle->get_Np_active());
      }
      particles.push_back(view);
    }
  }
};
} // namespace pic_ascent

#endif
