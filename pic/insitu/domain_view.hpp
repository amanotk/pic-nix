// -*- C++ -*-
#ifndef _PIC_INSITU_DOMAIN_VIEW_HPP_
#define _PIC_INSITU_DOMAIN_VIEW_HPP_

#include "../pic_chunk.hpp"
#include "nix/xtensor/field_layout.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace picnix::insitu
{
struct RawArrayView {
  float64*                       data = nullptr;
  std::vector<std::size_t>       shape;
  std::vector<std::size_t>       strides_bytes;
  std::vector<std::string_view>  components;
  std::vector<ComponentLocation> locations;
  std::string_view               location;
};

struct ParticleView {
  float64*                      data = nullptr;
  std::vector<std::size_t>      shape;
  std::vector<std::size_t>      strides_bytes;
  std::size_t                   np_active    = 0;
  std::size_t                   np_allocated = 0;
  float64                       charge       = 0.0;
  float64                       mass         = 0.0;
  std::vector<std::string_view> components;
  std::string_view              id_encoding;
};

struct DomainView {
  PicChunk*                 source    = nullptr;
  int                       domain_id = 0;
  int                       dimension = 0;
  int                       cycle     = 0;
  float64                   time      = 0.0;
  std::array<int, 3>        global_cell_shape_zyx{};
  std::array<int, 3>        local_cell_shape_zyx{};
  std::array<int, 3>        allocated_shape_zyx{};
  std::array<int, 3>        global_offset_zyx{};
  std::array<int, 3>        active_lower_zyx{};
  std::array<int, 3>        active_upper_zyx{};
  int                       ghost_width = 0;
  std::array<float64, 3>    spacing_xyz{};
  std::array<float64, 3>    physical_origin_xyz{};
  RawArrayView              uf;
  RawArrayView              uj;
  RawArrayView              um;
  RawArrayView              phi;
  std::vector<ParticleView> particles;

private:
  template <typename Array>
  static RawArrayView make_raw_view(Array& array)
  {
    RawArrayView result;
    result.data = array.data();
    for (auto extent : array.shape()) {
      result.shape.push_back(extent);
    }
    for (auto stride : array.strides()) {
      result.strides_bytes.push_back(stride * sizeof(float64));
    }
    return result;
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

    dimension = static_cast<int>(chunk.has_xdim()) + static_cast<int>(chunk.has_ydim()) +
                static_cast<int>(chunk.has_zdim());
    global_cell_shape_zyx = {global_extent(zrange.first, zrange.second, data.delz),
                             global_extent(yrange.first, yrange.second, data.dely),
                             global_extent(xrange.first, xrange.second, data.delx)};
    local_cell_shape_zyx  = {dims[0], dims[1], dims[2]};
    global_offset_zyx     = {offset[0], offset[1], offset[2]};
    active_lower_zyx      = {data.Lbz, data.Lby, data.Lbx};
    active_upper_zyx      = {data.Ubz, data.Uby, data.Ubx};
    allocated_shape_zyx   = {static_cast<int>(data.uf.shape(0)), static_cast<int>(data.uf.shape(1)),
                             static_cast<int>(data.uf.shape(2))};
    ghost_width           = data.boundary_margin;
    spacing_xyz           = {data.delx, data.dely, data.delz};
    physical_origin_xyz   = {data.xlim[0], data.ylim[0], data.zlim[0]};

    uf = make_raw_view(data.uf);
    uf.components.reserve(uf_components.size());
    uf.locations.reserve(uf_components.size());
    for (const auto& component : uf_components) {
      uf.components.push_back(component.name);
      uf.locations.push_back(component);
    }

    uj = make_raw_view(data.uj);
    uj.components.reserve(uj_components.size());
    uj.locations.reserve(uj_components.size());
    for (const auto& component : uj_components) {
      uj.components.push_back(component.name);
      uj.locations.push_back(component);
    }

    um = make_raw_view(data.um);
    um.components.assign(um_components.begin(), um_components.end());
    um.location = "cell";

    phi          = make_raw_view(data.phi);
    phi.location = "cell";

    particles.reserve(data.up.size());
    for (const auto& particle : data.up) {
      ParticleView view;
      view.components  = {"x", "y", "z", "ux", "uy", "uz", "id_bits"};
      view.id_encoding = "int64_bits_in_float64_slot";
      if (particle != nullptr) {
        view.data         = particle->xu.data();
        view.np_active    = static_cast<std::size_t>(particle->get_Np_active());
        view.np_allocated = static_cast<std::size_t>(particle->get_Np_total());
        view.charge       = particle->q;
        view.mass         = particle->m;
        for (auto extent : particle->xu.shape()) {
          view.shape.push_back(extent);
        }
        for (auto stride : particle->xu.strides()) {
          view.strides_bytes.push_back(stride * sizeof(float64));
        }
      }
      particles.push_back(std::move(view));
    }
  }
};
} // namespace picnix::insitu

#endif
