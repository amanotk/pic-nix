// -*- C++ -*-

#include "buffer.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix;

TEST_CASE("Buffer creation")
{
  Buffer buffer(1024);

  REQUIRE(buffer.size == 1024);
}

TEST_CASE("Buffer resize")
{
  Buffer buffer(1024);

  REQUIRE(buffer.size == 1024);

  buffer.resize(2048);
  REQUIRE(buffer.size == 2048);

  buffer.resize(512);
  REQUIRE(buffer.size == 512);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
