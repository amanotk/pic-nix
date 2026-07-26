// -*- C++ -*-
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

int main(int argc, char** argv)
{
  Catch::Session session;
  const int      return_code = session.applyCommandLine(argc, argv);

  if (return_code != 0) {
    return return_code;
  }
  return session.run();
}
