// -*- C++ -*-
#include "blueprint_builder.hpp"

#include "field_schema.hpp"
#include "nix/xtensor/xtensor_packer3d.hpp"

#include <conduit_blueprint.hpp>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pic_ascent
{
namespace
{
std::size_t element_count(const std::array<std::size_t, 3>& shape)
{
  return shape[0] * shape[1] * shape[2];
}

std::size_t element_count(const DomainView& view)
{
  return static_cast<std::size_t>(view.local_cell_shape_zyx[0]) *
         static_cast<std::size_t>(view.local_cell_shape_zyx[1]) *
         static_cast<std::size_t>(view.local_cell_shape_zyx[2]);
}

std::string indexed_name(const char* prefix, std::size_t index)
{
  std::ostringstream stream;
  stream << prefix << std::setfill('0') << std::setw(2) << index;
  return stream.str();
}

template <typename Names>
void add_strided_field(conduit::Node& domain, const std::string& field_name,
                       const std::string& topology, float64* values, std::size_t count,
                       std::size_t component_count, const Names& names,
                       std::size_t first_component = 0)
{
  auto& field          = domain["fields"][field_name];
  field["association"] = "element";
  field["topology"]    = topology;

  const auto stride = static_cast<conduit::index_t>(component_count * sizeof(float64));
  for (std::size_t component = 0; component < names.size(); component++) {
    const auto offset =
        static_cast<conduit::index_t>((first_component + component) * sizeof(float64));
    const auto dtype =
        conduit::DataType::float64(static_cast<conduit::index_t>(count), offset, stride);
    field["values"][std::string(names[component])].set_external(dtype, values);
  }
}

void add_owned_component(BlueprintPublication& publication, conduit::Node& domain,
                         const std::string& field_name, std::string_view component,
                         std::vector<float64>&& values)
{
  auto& buffer = publication.buffers.emplace_back(std::move(values));
  domain["fields"][field_name]["values"][std::string(component)].set_external(
      buffer.data(), static_cast<conduit::index_t>(buffer.size()));
}

void configure_cell_mesh(conduit::Node& domain, const DomainView& view)
{
  const int nz           = view.local_cell_shape_zyx[0];
  const int ny           = view.local_cell_shape_zyx[1];
  const int nx           = view.local_cell_shape_zyx[2];
  auto&     coordset     = domain["coordsets/cell_coords"];
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
  domain["topologies/cell_mesh/coordset"] = "cell_coords";
}

void configure_raw_mesh(conduit::Node& domain, const DomainView& view)
{
  const auto& shape      = view.uf.shape_zyx;
  auto&       coordset   = domain["coordsets/raw_storage_coords"];
  coordset["type"]       = "uniform";
  coordset["dims/i"]     = static_cast<conduit::index_t>(shape[2] + 1);
  coordset["origin/x"]   = 0.0;
  coordset["spacing/dx"] = 1.0;
  if (view.dimension >= 2) {
    coordset["dims/j"]     = static_cast<conduit::index_t>(shape[1] + 1);
    coordset["origin/y"]   = 0.0;
    coordset["spacing/dy"] = 1.0;
  }
  if (view.dimension >= 3) {
    coordset["dims/k"]     = static_cast<conduit::index_t>(shape[0] + 1);
    coordset["origin/z"]   = 0.0;
    coordset["spacing/dz"] = 1.0;
  }

  domain["topologies/raw_storage_mesh/type"]     = "uniform";
  domain["topologies/raw_storage_mesh/coordset"] = "raw_storage_coords";
}

void add_centered_fields(BlueprintPublication& publication, conduit::Node& domain,
                         const DomainView& view)
{
  auto data  = view.source->get_internal_data();
  auto field = nix::XtensorPacker3D::colocate_field(
      data.uf, data, view.active_axes_zyx[0], view.active_axes_zyx[1], view.active_axes_zyx[2]);

  constexpr std::array<std::string_view, 3> vector_components = {"x", "y", "z"};
  const auto                                count             = element_count(view);
  for (const auto& field_name : {std::string("E"), std::string("B")}) {
    domain["fields"][field_name]["association"] = "element";
    domain["fields"][field_name]["topology"]    = "cell_mesh";
  }
  for (std::size_t component = 0; component < 6; component++) {
    std::vector<float64> values;
    values.reserve(count);
    for (std::size_t iz = 0; iz < field.shape(0); iz++) {
      for (std::size_t iy = 0; iy < field.shape(1); iy++) {
        for (std::size_t ix = 0; ix < field.shape(2); ix++) {
          values.push_back(field(iz, iy, ix, component));
        }
      }
    }
    const auto field_name = component < 3 ? "E" : "B";
    add_owned_component(publication, domain, field_name, vector_components[component % 3],
                        std::move(values));
  }

  const std::size_t species_count = data.um.shape(3);
  for (std::size_t species = 0; species < species_count; species++) {
    const auto field_name                       = indexed_name("um", species);
    domain["fields"][field_name]["association"] = "element";
    domain["fields"][field_name]["topology"]    = "cell_mesh";
    for (std::size_t component = 0; component < moment_components.size(); component++) {
      std::vector<float64> values;
      values.reserve(count);
      for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
        for (int iy = data.Lby; iy <= data.Uby; iy++) {
          for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
            values.push_back(data.um(iz, iy, ix, species, component));
          }
        }
      }
      add_owned_component(publication, domain, field_name, moment_components[component],
                          std::move(values));
    }
  }
}

void add_raw_data(conduit::Node& domain, const DomainView& view)
{
  if (view.uf.shape_zyx != view.uj.shape_zyx) {
    throw std::runtime_error("PIC raw uf and uj shapes do not match");
  }

  configure_raw_mesh(domain, view);
  const auto count = element_count(view.uf.shape_zyx);
  add_strided_field(domain, "uf", "raw_storage_mesh", view.uf.data, count, view.uf.component_count,
                    uf_components);
  add_strided_field(domain, "uj", "raw_storage_mesh", view.uj.data, count, view.uj.component_count,
                    uj_components);
  domain["pic/neighbors/domain_ids"].set(view.neighbor_domain_ids.data(),
                                         view.neighbor_domain_ids.size());
  domain["pic/neighbors/neighbor_ranks"].set(view.neighbor_ranks.data(),
                                             view.neighbor_ranks.size());
}

void add_particles(conduit::Node& domain, const DomainView& view)
{
  for (std::size_t species = 0; species < view.particles.size(); species++) {
    const auto& particle = view.particles[species];
    if (particle.data == nullptr || particle.np_active == 0) {
      continue;
    }
    const auto count = static_cast<conduit::index_t>(particle.np_active * 7);
    domain["pic/particles"][indexed_name("particle", species)]["xu"].set_external(particle.data,
                                                                                  count);
  }
}
} // namespace

BlueprintPublication BlueprintBuilder::build(const std::vector<PicChunk*>& chunks, int cycle,
                                             float64 time, const json& configuration,
                                             const BlueprintOptions& options)
{
  BlueprintPublication publication;
  if (chunks.empty()) {
    return publication;
  }
  if (!options.centered && !options.raw) {
    throw std::invalid_argument("PIC Ascent publication requires a Blueprint mesh");
  }

  if (chunks.front() == nullptr) {
    throw std::invalid_argument("PIC Ascent publication received a null chunk");
  }
  const int boundary_margin = chunks.front()->get_boundary_margin();

  for (auto* chunk : chunks) {
    if (chunk == nullptr) {
      throw std::invalid_argument("PIC Ascent publication received a null chunk");
    }
    if (chunk->get_boundary_margin() != boundary_margin) {
      throw std::invalid_argument("PIC Ascent publication requires a common boundary margin");
    }

    DomainView view(*chunk, cycle, time);
    auto&      domain         = publication.node["domain_" + std::to_string(view.domain_id)];
    domain["state/domain_id"] = view.domain_id;
    domain["state/cycle"]     = cycle;
    domain["state/time"]      = time;

    if (options.centered) {
      configure_cell_mesh(domain, view);
      add_centered_fields(publication, domain, view);
    }
    if (options.raw) {
      add_raw_data(domain, view);
    }
    if (options.particles) {
      add_particles(domain, view);
    }
  }

  auto& owner = publication.node["domain_" + std::to_string(chunks.front()->get_id())];
  owner["pic/schema_version"]  = schema_version;
  owner["pic/boundary_margin"] = boundary_margin;
  owner["pic/config"]          = configuration.dump();

  return publication;
}
} // namespace pic_ascent
