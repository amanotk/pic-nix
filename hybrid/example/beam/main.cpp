// -*- C++ -*-
#include "hybrid_application.hpp"

#include <memory>

int main(int argc, char** argv)
{
  auto                      interface = std::make_shared<hybrid::HybridApplicationInterface>();
  hybrid::HybridApplication app(argc, argv, interface);
  return app.main();
}
