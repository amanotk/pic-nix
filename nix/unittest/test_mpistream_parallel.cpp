// -*- C++ -*-

#include "mpistream.hpp"

#include "catch.hpp"

bool require_mpi_size(int expected);

TEST_CASE("recursively_create_directory", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }
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

    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = status;
    if (thisrank == 0) {
      for (int i = 0; i < dirnames.size(); i++) {
        ok = ok && std::filesystem::exists(dirnames[i]);
      }
    }
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    REQUIRE(ok_int == 1);
  }

  SECTION("two level2")
  {
    std::vector<std::string> dirnames = {"foo/000000/000000", "foo/000000/000002",
                                         "foo/000004/000004", "foo/000004/000006"};

    bool status = MpiStream::create_directory_tree("foo", thisrank, nprocess, 2);

    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = status;
    if (thisrank == 0) {
      for (int i = 0; i < dirnames.size(); i++) {
        ok = ok && std::filesystem::exists(dirnames[i]);
      }
    }
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    REQUIRE(ok_int == 1);
  }

  // cleanup
  if (thisrank == 0 && std::filesystem::exists("foo")) {
    std::filesystem::remove_all("foo");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
