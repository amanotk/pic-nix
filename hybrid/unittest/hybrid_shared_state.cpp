// -*- C++ -*-
#include "hybrid_application.hpp"
#include "hybrid_chunk.hpp"

#include "example/beam/beam_chunk.hpp"

#include "engine/field.hpp"
#include "engine/moment.hpp"
#include "nix/cfgparser.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{
class BeamSharedInterface : public hybrid::HybridApplicationInterface
{
public:
  nix::Application::PtrChunk create_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id) override
  {
    return std::make_unique<hybrid::beam::BeamChunk>(dims, has_dim, id);
  }
};

class SharedStateApplication final : public hybrid::HybridApplication
{
public:
  using HybridApplication::HybridApplication;

protected:
  void pcc2_stage_completed(hybrid::engine::Pcc2Stage stage) override
  {
    if (!hybrid::engine::pcc2_is_field_stage(stage)) {
      return;
    }
    for (const auto& chunk_ptr : chunkvec) {
      auto data = static_cast<hybrid::HybridChunk&>(*chunk_ptr).get_internal_data();
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        for (int component = 0; component < hybrid::num_field_components; ++component) {
          const auto reference = data.work_field_cell(data.Lbz, data.Lby, ix, component);
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              if (std::abs(data.work_field_cell(iz, iy, ix, component) - reference) > 1.0e-12) {
                throw std::runtime_error(
                    std::string("Shared Hybrid transverse symmetry fails after ") +
                    hybrid::engine::pcc2_stage_name(stage));
              }
            }
          }
        }
      }
    }
  }

  void setup_chunks() override
  {
    HybridApplication::setup_chunks();
    if (nprocess != 1 || chunkvec.size() != 1) {
      throw std::runtime_error("Shared Hybrid state requires one MPI rank and one chunk");
    }

    auto& chunk = static_cast<hybrid::HybridChunk&>(*chunkvec.front());
    auto  data  = chunk.get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int component = 0; component < hybrid::num_field_components; ++component) {
            if (!std::isfinite(data.field_cell(iz, iy, ix, component))) {
              throw std::runtime_error("Shared Hybrid initial field is non-finite");
            }
          }
          for (int component = 0; component < hybrid::num_fluid_components; ++component) {
            if (!std::isfinite(data.fluid(iz, iy, ix, component))) {
              throw std::runtime_error("Shared Hybrid initial fluid is non-finite");
            }
          }
          for (int species = 0; species < data.num_species; ++species) {
            for (int component = 0; component < hybrid::num_moment_components; ++component) {
              if (!std::isfinite(data.moment_kinetic(iz, iy, ix, species, component))) {
                throw std::runtime_error("Shared Hybrid initial moment is non-finite");
              }
            }
          }
        }
      }
    }

    for (int species = 0; species < data.num_species; ++species) {
      auto& particle = *data.particles[species];
      if (particle.Np <= 0) {
        throw std::runtime_error("Shared Hybrid state has no particles for species " +
                                 std::to_string(species));
      }
    }
  }

  void push() override
  {
    HybridApplication::push();
    auto& chunk = static_cast<hybrid::HybridChunk&>(*chunkvec.front());
    auto  data  = chunk.get_internal_data();
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int component = 0; component < hybrid::num_field_components; ++component) {
            if (!std::isfinite(data.field_cell(iz, iy, ix, component))) {
              throw std::runtime_error("Shared Hybrid post-push field is non-finite");
            }
          }
        }
      }
    }
  }
};
} // namespace

int main(int argc, char** argv)
{
  auto                   interface = std::make_shared<BeamSharedInterface>();
  SharedStateApplication application(argc, argv, interface);
  return application.main();
}
