// -*- C++ -*-
#include "beam_chunk.hpp"
#include "hybrid_application.hpp"

#include <memory>

namespace
{
class BeamInterface : public hybrid::HybridApplicationInterface
{
public:
  nix::Application::PtrChunk create_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id) override
  {
    return std::make_unique<hybrid::beam::BeamChunk>(dims, has_dim, id);
  }
};
} // namespace

int main(int argc, char** argv)
{
  auto                      interface = std::make_shared<BeamInterface>();
  hybrid::HybridApplication app(argc, argv, interface);
  return app.main();
}
