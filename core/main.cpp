// -*- C++ -*-

#include "expic3d.hpp"

constexpr int order = 1;

class MainChunk;
class MainApplication;

class MainChunk : public ExChunk3D<order>
{
public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  virtual void setup(json& config) override
  {
    ExChunk3D<order>::setup(config);
  }
};

class MainApplication : public ExPIC3D<order>
{
public:
  using ExPIC3D<order>::ExPIC3D; // inherit constructors

  std::unique_ptr<ExChunk3D<order>> create_chunk(const int dims[], const int id) override
  {
    return std::make_unique<MainChunk>(dims, id);
  }
};

//
// main
//
int main(int argc, char** argv)
{
  MainApplication app(argc, argv);
  return app.main(std::cout);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
