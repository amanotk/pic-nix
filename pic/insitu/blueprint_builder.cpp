// -*- C++ -*-
#include "blueprint_builder.hpp"

#include "nix/xtensor/xtensor_packer3d.hpp"

#include <conduit_blueprint.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>

namespace picnix::insitu
{
namespace
{
inline void append_value(conduit::Node& node, std::string_view value)
{
  node.append().set(std::string(value));
}

template <typename T>
void append_value(conduit::Node& node, const T& value)
{
  node.append().set(value);
}

template <typename Range>
void append_values(conduit::Node& node, const Range& values)
{
  for (const auto value : values) {
    append_value(node, value);
  }
}

std::size_t element_count(const DomainView& view)
{
  return static_cast<std::size_t>(view.local_cell_shape_zyx[0]) *
         static_cast<std::size_t>(view.local_cell_shape_zyx[1]) *
         static_cast<std::size_t>(view.local_cell_shape_zyx[2]);
}

void append_locations(conduit::Node& node, const std::vector<ComponentLocation>& locations)
{
  for (const auto& location : locations) {
    auto& entry          = node.append();
    entry["name"]        = std::string(location.name);
    entry["association"] = std::string(location.association);
    append_values(entry["normalized_xyz"], location.normalized_xyz);
  }
}

void add_raw_array(conduit::Node& domain, const char* name, const RawArrayView& view)
{
  auto&             raw   = domain[std::string("picnix/raw/") + name];
  const std::size_t count = std::accumulate(view.shape.begin(), view.shape.end(), std::size_t{1},
                                            std::multiplies<std::size_t>());
  if (view.data != nullptr && count > 0) {
    raw["values"].set_external(view.data, static_cast<conduit::index_t>(count));
  }
  append_values(raw["shape"], view.shape);
  append_values(raw["strides_bytes"], view.strides_bytes);
  append_values(raw["components"], view.components);
  if (!view.locations.empty()) {
    append_locations(raw["component_locations"], view.locations);
  } else if (!view.location.empty()) {
    raw["location"] = std::string(view.location);
  }
}

void add_raw_particles(conduit::Node& domain, const DomainView& view)
{
  for (std::size_t species = 0; species < view.particles.size(); species++) {
    const auto&       particle     = view.particles[species];
    const std::string species_name = std::string("species_") +
                                     (species < 10    ? "00"
                                      : species < 100 ? "0"
                                                      : "") +
                                     std::to_string(species);
    auto&             raw   = domain["picnix/particles"][species_name];
    const std::size_t count = std::accumulate(particle.shape.begin(), particle.shape.end(),
                                              std::size_t{1}, std::multiplies<std::size_t>());
    if (particle.data != nullptr && count > 0) {
      raw["values"].set_external(particle.data, static_cast<conduit::index_t>(count));
    }
    append_values(raw["shape"], particle.shape);
    append_values(raw["strides_bytes"], particle.strides_bytes);
    raw["np_active"]      = static_cast<conduit::index_t>(particle.np_active);
    raw["np_allocated"]   = static_cast<conduit::index_t>(particle.np_allocated);
    raw["particle_width"] = 7;
    raw["charge"]         = particle.charge;
    raw["mass"]           = particle.mass;
    append_values(raw["components"], particle.components);
    raw["id_encoding"] = std::string(particle.id_encoding);
  }
}

void add_field(conduit::Node& domain, const char* name, const char* topology,
               const char* association, std::vector<float64>& values)
{
  auto& field          = domain["fields"][name];
  field["association"] = association;
  field["topology"]    = topology;
  field["values"].set_external(values.data(), static_cast<conduit::index_t>(values.size()));
}

void add_vector_component(conduit::Node& domain, const char* field_name, const char* component,
                          std::vector<float64>& values)
{
  domain["fields"][field_name]["values"][component].set_external(
      values.data(), static_cast<conduit::index_t>(values.size()));
}

template <typename Array>
std::vector<float64> copy_component(const Array& values, std::size_t component, std::size_t count)
{
  std::vector<float64> result;
  result.reserve(count);
  for (std::size_t iz = 0; iz < values.shape(0); iz++) {
    for (std::size_t iy = 0; iy < values.shape(1); iy++) {
      for (std::size_t ix = 0; ix < values.shape(2); ix++) {
        result.push_back(values(iz, iy, ix, component));
      }
    }
  }
  return result;
}

void configure_mesh(conduit::Node& domain, const DomainView& view)
{
  const int nz           = view.local_cell_shape_zyx[0];
  const int ny           = view.local_cell_shape_zyx[1];
  const int nx           = view.local_cell_shape_zyx[2];
  auto&     coordset     = domain["coordsets/cell_vertices"];
  coordset["type"]       = "uniform";
  coordset["dims/i"]     = nx + 1;
  coordset["origin/x"]   = view.physical_origin_xyz[0];
  coordset["spacing/dx"] = view.spacing_xyz[0];
  if (view.dimension >= 2) {
    coordset["dims/j"]     = ny + 1;
    coordset["origin/y"]   = view.physical_origin_xyz[1];
    coordset["spacing/dy"] = view.spacing_xyz[1];
  }
  if (view.dimension >= 3) {
    coordset["dims/k"]     = nz + 1;
    coordset["origin/z"]   = view.physical_origin_xyz[2];
    coordset["spacing/dz"] = view.spacing_xyz[2];
  }

  domain["topologies/cell_mesh/type"]     = "uniform";
  domain["topologies/cell_mesh/coordset"] = "cell_vertices";
}
} // namespace

BlueprintPublication BlueprintBuilder::build(const std::vector<PicChunk*>& chunks, int cycle,
                                             float64 time, const BlueprintOptions& options)
{
  BlueprintPublication publication;
  publication.buffers.reserve(chunks.size() * 20);

  for (auto* chunk : chunks) {
    DomainView view(*chunk, cycle, time);
    auto&      domain = publication.node["domain_" + std::to_string(view.domain_id)];
    const auto count  = element_count(view);

    domain["state/domain_id"] = view.domain_id;
    domain["state/cycle"]     = cycle;
    domain["state/time"]      = time;
    configure_mesh(domain, view);

    domain["picnix/schema_version"] = raw_schema_version;
    domain["picnix/mesh/dimension"] = view.dimension;
    append_values(domain["picnix/mesh/global_cell_shape"], view.global_cell_shape_zyx);
    append_values(domain["picnix/mesh/local_cell_shape"], view.local_cell_shape_zyx);
    append_values(domain["picnix/mesh/allocated_shape"], view.allocated_shape_zyx);
    append_values(domain["picnix/mesh/global_offset"], view.global_offset_zyx);
    append_values(domain["picnix/mesh/active_lower"], view.active_lower_zyx);
    append_values(domain["picnix/mesh/active_upper"], view.active_upper_zyx);
    domain["picnix/mesh/ghost_width"] = view.ghost_width;
    append_values(domain["picnix/mesh/spacing"], view.spacing_xyz);
    append_values(domain["picnix/mesh/physical_origin"], view.physical_origin_xyz);
    domain["picnix/mesh/layout"] = "C/zyx-components-last";

    if (options.raw) {
      add_raw_array(domain, "uf", view.uf);
      add_raw_array(domain, "uj", view.uj);
      add_raw_array(domain, "um", view.um);
      add_raw_array(domain, "phi", view.phi);
      if (options.particles) {
        add_raw_particles(domain, view);
      }
    }

    if (options.centered) {
      auto data = view.source->get_internal_data();
      auto e    = nix::XtensorPacker3D::colocate_field(data.uf, data);
      auto j    = nix::XtensorPacker3D::colocate_current(data.uj, data);

      std::array<std::vector<float64>, 6> e_components;
      std::array<std::vector<float64>, 4> j_components;
      for (std::size_t component = 0; component < e_components.size(); component++) {
        e_components[component] = copy_component(e, component, count);
      }
      for (std::size_t component = 0; component < j_components.size(); component++) {
        j_components[component] = copy_component(j, component, count);
      }

      for (auto& values : e_components) {
        publication.buffers.push_back(std::move(values));
      }
      for (auto& values : j_components) {
        publication.buffers.push_back(std::move(values));
      }

      auto& e_field          = domain["fields/E"];
      e_field["association"] = "element";
      e_field["topology"]    = "cell_mesh";
      add_vector_component(domain, "E", "x", publication.buffers[publication.buffers.size() - 10]);
      add_vector_component(domain, "E", "y", publication.buffers[publication.buffers.size() - 9]);
      add_vector_component(domain, "E", "z", publication.buffers[publication.buffers.size() - 8]);

      auto& b_field          = domain["fields/B"];
      b_field["association"] = "element";
      b_field["topology"]    = "cell_mesh";
      add_vector_component(domain, "B", "x", publication.buffers[publication.buffers.size() - 7]);
      add_vector_component(domain, "B", "y", publication.buffers[publication.buffers.size() - 6]);
      add_vector_component(domain, "B", "z", publication.buffers[publication.buffers.size() - 5]);

      auto& j_field          = domain["fields/J"];
      j_field["association"] = "element";
      j_field["topology"]    = "cell_mesh";
      add_vector_component(domain, "J", "x", publication.buffers[publication.buffers.size() - 3]);
      add_vector_component(domain, "J", "y", publication.buffers[publication.buffers.size() - 2]);
      add_vector_component(domain, "J", "z", publication.buffers[publication.buffers.size() - 1]);
      add_field(domain, "rho", "cell_mesh", "element",
                publication.buffers[publication.buffers.size() - 4]);

      auto phi_values = std::vector<float64>();
      phi_values.reserve(count);
      for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
        for (int iy = data.Lby; iy <= data.Uby; iy++) {
          for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
            phi_values.push_back(data.phi(iz, iy, ix));
          }
        }
      }
      publication.buffers.push_back(std::move(phi_values));
      add_field(domain, "phi", "cell_mesh", "element", publication.buffers.back());
    }
  }

  return publication;
}
} // namespace picnix::insitu
