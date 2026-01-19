// -*- C++ -*-


#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

int main(int argc, char** argv)
{
  using namespace Catch::Clara;

  // catch
  Catch::Session session;

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
    return returnCode;
  }

  int result = session.run();

  return result;
}
