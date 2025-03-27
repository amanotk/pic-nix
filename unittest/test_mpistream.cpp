// -*- C++ -*-

#include "mpistream.hpp"

#include "catch.hpp"

TEST_CASE("get_filename_pattern")
{
  SECTION("single level")
  {
    REQUIRE(MpiStream::get_filename_pattern(0, 16) == "000000");
    REQUIRE(MpiStream::get_filename_pattern(0, 16, -1) == "000000");
    REQUIRE(MpiStream::get_filename_pattern(1, 16, 24) == "000001");
  }

  SECTION("two levels")
  {
    REQUIRE(MpiStream::get_filename_pattern(0, 8, 4) == "000000/000000");
    REQUIRE(MpiStream::get_filename_pattern(1, 8, 4) == "000000/000001");
    REQUIRE(MpiStream::get_filename_pattern(2, 8, 4) == "000000/000002");
    REQUIRE(MpiStream::get_filename_pattern(3, 8, 4) == "000000/000003");
    REQUIRE(MpiStream::get_filename_pattern(4, 8, 4) == "000004/000004");
    REQUIRE(MpiStream::get_filename_pattern(5, 8, 4) == "000004/000005");
    REQUIRE(MpiStream::get_filename_pattern(6, 8, 4) == "000004/000006");
    REQUIRE(MpiStream::get_filename_pattern(7, 8, 4) == "000004/000007");

    REQUIRE(MpiStream::get_filename_pattern(0, 16, 4) == "000000/000000");
    REQUIRE(MpiStream::get_filename_pattern(4, 16, 4) == "000004/000004");
    REQUIRE(MpiStream::get_filename_pattern(8, 16, 4) == "000008/000008");
    REQUIRE(MpiStream::get_filename_pattern(12, 16, 4) == "000012/000012");
  }

  SECTION("three levels")
  {
    REQUIRE(MpiStream::get_filename_pattern(0, 32, 4) == "000000/000000/000000");
    REQUIRE(MpiStream::get_filename_pattern(4, 32, 4) == "000000/000004/000004");
    REQUIRE(MpiStream::get_filename_pattern(8, 32, 4) == "000000/000008/000008");
    REQUIRE(MpiStream::get_filename_pattern(12, 32, 4) == "000000/000012/000012");
    REQUIRE(MpiStream::get_filename_pattern(16, 32, 4) == "000016/000016/000016");
    REQUIRE(MpiStream::get_filename_pattern(20, 32, 4) == "000016/000020/000020");
    REQUIRE(MpiStream::get_filename_pattern(24, 32, 4) == "000016/000024/000024");
    REQUIRE(MpiStream::get_filename_pattern(28, 32, 4) == "000016/000028/000028");
  }
}

TEST_CASE("get_filename")
{
  SECTION("generic")
  {
    REQUIRE(MpiStream::get_filename("tmp", ".out", 0, 16) == "tmp/000000.out");
    REQUIRE(MpiStream::get_filename("tmp", ".err", 1, 16) == "tmp/000001.err");
    REQUIRE(MpiStream::get_filename("tmp", ".out", 0, 8, 4) == "tmp/000000/000000.out");
    REQUIRE(MpiStream::get_filename("tmp", ".out", 4, 8, 4) == "tmp/000004/000004.out");
  }

  SECTION("stdout/tderr")
  {
    REQUIRE(MpiStream::get_stdout_filename(".", 0, 16) == "./000000.stdout");
    REQUIRE(MpiStream::get_stderr_filename(".", 1, 16) == "./000001.stderr");
    REQUIRE(MpiStream::get_stdout_filename("", 0, 16) == "/dev/null");
    REQUIRE(MpiStream::get_stderr_filename("", 0, 16) == "/dev/null");
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
