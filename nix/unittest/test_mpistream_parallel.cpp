// -*- C++ -*-

#include "mpistream.hpp"

#include "catch.hpp"

TEST_CASE("recursively_create_directory")
{
  int thisrank = 0;
  int nprocess = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

  // remove directory if exists
  if (thisrank == 0 && std::filesystem::exists("foo")) {
    std::filesystem::remove_all("foo");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  REQUIRE(nprocess == 8);

  SECTION("single level")
  {
    std::vector<std::string> dirnames = {"foo/000000", "foo/000004"};

    bool status = MpiStream::create_directory_tree("foo", thisrank, nprocess, 4);

    REQUIRE(status == true);
    for (int i = 0; i < dirnames.size(); i++) {
      REQUIRE(std::filesystem::exists(dirnames[i]) == true);
    }
  }

  SECTION("two level2")
  {
    std::vector<std::string> dirnames = {"foo/000000/000000", "foo/000000/000002",
                                         "foo/000004/000004", "foo/000004/000006"};

    bool status = MpiStream::create_directory_tree("foo", thisrank, nprocess, 2);

    REQUIRE(status == true);
    for (int i = 0; i < dirnames.size(); i++) {
      REQUIRE(std::filesystem::exists(dirnames[i]) == true);
    }
  }

  // cleanup
  if (thisrank == 0 && std::filesystem::exists("foo")) {
    std::filesystem::remove_all("foo");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
