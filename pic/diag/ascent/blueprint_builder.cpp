// -*- C++ -*-
#include "blueprint_builder.hpp"

#include "field_schema.hpp"
#include "nix/xtensor/xtensor_packer3d.hpp"

#include <cstddef>
#include <iomanip>
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

void add_owned_component(BlueprintPublication& publication, conduit::Node& domain,
                         const std::string& field_name, std::string_view component,
                         std::vector<float64>&& values)
{
  auto& buffer = publication.buffers.emplace_back(std::move(values));
  domain["fields"][field_name]["values"][std::string(component)].set_external(
      buffer.data(), static_cast<conduit::index_t>(buffer.size()));
}

void add_owned_scalar(BlueprintPublication& publication, conduit::Node& domain,
                      const std::string& field_name, std::vector<float64>&& values)
{
  auto& buffer = publication.buffers.emplace_back(std::move(values));
  domain["fields"][field_name]["values"].set_external(buffer.data(),
                                                      static_cast<conduit::index_t>(buffer.size()));
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

void add_centered_fields(BlueprintPublication& publication, conduit::Node& domain,
                         const DomainView& view, const BlueprintOptions& options)
{
  auto                                      data              = view.source->get_internal_data();
  constexpr std::array<std::string_view, 3> vector_components = {"x", "y", "z"};
  const auto                                count             = element_count(view);

  if (options.electric_field || options.magnetic_field) {
    auto field = nix::XtensorPacker3D::colocate_field(
        data.uf, data, view.active_axes_zyx[0], view.active_axes_zyx[1], view.active_axes_zyx[2]);
    for (std::size_t component = 0; component < 6; component++) {
      const bool selected = component < 3 ? options.electric_field : options.magnetic_field;
      if (!selected) {
        continue;
      }
      std::vector<float64> values;
      values.reserve(count);
      for (std::size_t iz = 0; iz < field.shape(0); iz++) {
        for (std::size_t iy = 0; iy < field.shape(1); iy++) {
          for (std::size_t ix = 0; ix < field.shape(2); ix++) {
            values.push_back(field(iz, iy, ix, component));
          }
        }
      }
      const auto field_name                       = component < 3 ? "E" : "B";
      domain["fields"][field_name]["association"] = "element";
      domain["fields"][field_name]["topology"]    = "cell_mesh";
      add_owned_component(publication, domain, field_name, vector_components[component % 3],
                          std::move(values));
    }
  }

  if (!options.mass_current && !options.energy_momentum) {
    return;
  }

  const std::size_t species_count = data.um.shape(3);
  for (std::size_t species = 0; species < species_count; species++) {
    for (std::size_t component = 0; component < moment_components.size(); component++) {
      const bool selected =
          component < mass_current_component_count ? options.mass_current : options.energy_momentum;
      if (!selected) {
        continue;
      }
      std::vector<float64> values;
      values.reserve(count);
      for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
        for (int iy = data.Lby; iy <= data.Uby; iy++) {
          for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
            values.push_back(data.um(iz, iy, ix, species, component));
          }
        }
      }
      const auto field_name =
          indexed_name("um", species) + "_" + std::string(moment_components[component]);
      domain["fields"][field_name]["association"] = "element";
      domain["fields"][field_name]["topology"]    = "cell_mesh";
      add_owned_scalar(publication, domain, field_name, std::move(values));
    }
  }
}

void add_raw_arrays(conduit::Node& domain, const DomainView& view)
{
  if (view.uf.shape_zyx != view.uj.shape_zyx) {
    throw std::runtime_error("PIC raw uf and uj shapes do not match");
  }

  if (view.uf.component_count != 6 || view.uj.component_count != 4) {
    throw std::runtime_error("PIC raw uf and uj component counts must be 6 and 4");
  }

  const std::array<conduit::int64, 3> shape = {
      static_cast<conduit::int64>(view.uf.shape_zyx[0]),
      static_cast<conduit::int64>(view.uf.shape_zyx[1]),
      static_cast<conduit::int64>(view.uf.shape_zyx[2]),
  };
  domain["pic/raw/shape"].set(shape.data(), shape.size());
  const auto count = element_count(view.uf.shape_zyx);
  domain["pic/raw/uf"].set_external(view.uf.data,
                                    static_cast<conduit::index_t>(count * view.uf.component_count));
  domain["pic/raw/uj"].set_external(view.uj.data,
                                    static_cast<conduit::index_t>(count * view.uj.component_count));
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

    configure_cell_mesh(domain, view);
    add_centered_fields(publication, domain, view, options);
    if (options.raw_fields) {
      add_raw_arrays(domain, view);
    }
    if (options.raw_particles) {
      add_particles(domain, view);
    }
  }

  auto& owner = publication.node["domain_" + std::to_string(chunks.front()->get_id())];
  owner["pic/boundary_margin"] = boundary_margin;
  owner["pic/config"]          = configuration.dump();

  return publication;
}
} // namespace pic_ascent
