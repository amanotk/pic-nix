// -*- C++ -*-

#include "chunkmap.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "catch.hpp"

using namespace nix;

class ChunkMapTest : public ChunkMap
{
public:
  using ChunkMap::ChunkMap;
  using ChunkMap::dims;
  using ChunkMap::periodicity;

  void test_dimension(int nz, int ny, int nx)
  {
    REQUIRE(dims[0] == nz);
    REQUIRE(dims[1] == ny);
    REQUIRE(dims[2] == nx);
  }

  void test_periodicity(int pz, int py, int px)
  {
    REQUIRE(periodicity[0] == pz);
    REQUIRE(periodicity[1] == py);
    REQUIRE(periodicity[2] == px);
  }

  void test_get_neighbor_coord(int coord, int delta, int dir)
  {
    int first_coord = 0;
    int last_coord  = dims[dir] - 1;
    int nb_coord    = this->get_neighbor_coord(coord, delta, dir);

    if (periodicity[dir] == 1) {
      if (coord == first_coord && delta == -1) {
        REQUIRE(nb_coord == last_coord);
        return;
      }

      if (coord == last_coord && delta == +1) {
        REQUIRE(nb_coord == first_coord);
        return;
      }
    } else {
      if (coord == first_coord && delta == -1) {
        REQUIRE(nb_coord == -1);
        return;
      }

      if (coord == last_coord && delta == +1) {
        REQUIRE(nb_coord == -1);
        return;
      }
    }

    REQUIRE(nb_coord == coord + delta);
  }

  void test_set_rank_boundary(std::vector<int>& boundary)
  {
    this->set_rank_boundary(boundary);

    for (int i = 0; i < this->size; i++) {
      int rank = this->get_rank(i);
      REQUIRE(boundary[rank] <= i);
      REQUIRE(boundary[rank + 1] > i);
    }
  }

  void test_get_rank_boundary()
  {
    std::vector<int> boundary = this->get_rank_boundary();

    for (int i = 0; i < this->size; i++) {
      int rank = this->get_rank(i);
      REQUIRE(boundary[rank] <= i);
      REQUIRE(boundary[rank + 1] > i);
    }
  }
};

TEST_CASE("Initialization")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("3D")
  {
    ChunkMapTest chunkmap(Cz, Cy, Cx);
    REQUIRE(chunkmap.validate());
  }
}

TEST_CASE("Dimension")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("3D")
  {
    ChunkMapTest chunkmap(Cz, Cy, Cx);
    chunkmap.test_dimension(Cz, Cy, Cx);
  }
}

TEST_CASE("Periodicity")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("3D")
  {
    ChunkMapTest chunkmap(Cz, Cy, Cx);
    chunkmap.test_periodicity(1, 1, 1);

    chunkmap.set_periodicity(1, 1, 0);
    chunkmap.test_periodicity(1, 1, 0);
  }
}

TEST_CASE("set_rank from boundary array")
{
  SECTION("3D")
  {
    int Cx = 4;
    int Cy = 2;
    int Cz = 2;
    int Nc = Cz * Cy * Cx;

    std::vector<int> boundary{0, 4, 9, 14, Nc};
    ChunkMapTest     chunkmap(Cz, Cy, Cx);

    chunkmap.test_set_rank_boundary(boundary);
  }
}

TEST_CASE("get_rank_boundary")
{
  const int        nproc    = 4;
  std::vector<int> boundary = {0, 4, 9, 14, 16};

  SECTION("3D")
  {
    int Cx = 4;
    int Cy = 2;
    int Cz = 2;
    int Nc = Cz * Cy * Cx;

    std::vector<int> boundary{0, 4, 9, 14, Nc};
    ChunkMapTest     chunkmap(Cz, Cy, Cx);

    chunkmap.set_rank_boundary(boundary);
    chunkmap.test_get_rank_boundary();
  }
}

TEST_CASE("get_neighbor_coord")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  ChunkMapTest chunkmap(Cz, Cy, Cx);

  SECTION("X")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 2);
    chunkmap.test_get_neighbor_coord(0, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx / 2, +1, 2);
    chunkmap.test_get_neighbor_coord(Cx / 2, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, +1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, -1, 2);

    // non-periodic
    chunkmap.set_periodicity(0, 1, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, +1, 2);
  }

  SECTION("Y")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy / 2, +1, 1);
    chunkmap.test_get_neighbor_coord(Cy / 2, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, +1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, -1, 1);

    // non-periodic
    chunkmap.set_periodicity(1, 0, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, +1, 1);
  }

  SECTION("Z")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 0);
    chunkmap.test_get_neighbor_coord(0, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz / 2, +1, 0);
    chunkmap.test_get_neighbor_coord(Cz / 2, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, +1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, -1, 0);

    // non-periodic
    chunkmap.set_periodicity(1, 1, 0);
    chunkmap.test_get_neighbor_coord(0, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, +1, 0);
  }
}

TEST_CASE("Save to and load from file")
{
  const std::string filename = "test_chunkmap.json";

  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("3D")
  {
    ChunkMapTest chunkmap(Cz, Cy, Cx);

    // save
    {
      auto obj = chunkmap.to_json();

      std::ofstream ofs(filename);
      ofs << std::setw(2) << obj;
    }

    // load
    {
      json obj;

      std::ifstream ifs(filename);
      ifs >> obj;

      chunkmap.from_json(obj);
    }

    // check for load
    REQUIRE(chunkmap.validate());

    // cleanup
    std::remove(filename.c_str());
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
