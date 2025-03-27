// -*- C++ -*-

#include <iostream>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char** argv)
{
  using namespace Catch::clara;

  // catch
  Catch::Session session;

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
    return returnCode;
  }

  int result = session.run();

  return result;
}
