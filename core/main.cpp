// -*- C++ -*-

#include "expic3d.hpp"

using App = ExPIC3D<1>;

//
// main
//
int main(int argc, char **argv)
{
  App app(argc, argv);
  return app.main(std::cout);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
