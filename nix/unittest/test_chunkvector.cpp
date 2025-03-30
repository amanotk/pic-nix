// -*- C++ -*-
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "chunkvector.hpp"

#include "catch.hpp"

using namespace nix;

class MockChunkMap : public ChunkMap
{
private:
  int dims[3];

public:
  MockChunkMap(int cz, int cy, int cx) : ChunkMap(cz, cy, cx)
  {
  }

  virtual int get_chunkid(int cz, int cy, int cx) override
  {
    return 0;
  }

  virtual int get_rank(int id) override
  {
    return 0;
  }

  virtual int get_neighbor_coord(int coord, int delta, int dir) override
  {
    return 0;
  }

  virtual void get_coordinate(int id, int& cz, int& cy, int& cx) override
  {
    cx = 0;
    cy = 0;
    cz = 0;
  }
};

class MockChunk
{
private:
  int myid;

public:
  MockChunk(int id = 0) : myid(id)
  {
  }

  int get_id() const
  {
    return myid;
  }

  void set_nb_id(int dirz, int diry, int dirx, int id)
  {
  }

  void set_nb_rank(int dirz, int diry, int dirx, int rank)
  {
  }
};

class ChunkVectorTest : public ChunkVector<std::unique_ptr<MockChunk>>
{
public:
  bool is_sorted()
  {
    return std::is_sorted(this->begin(), this->end(),
                          [](const auto& x, const auto& y) { return x->get_id() < y->get_id(); });
  }
};

TEST_CASE("sort_and_shrink")
{
  using MockChunkVector = std::vector<std::unique_ptr<MockChunk>>;
  const int size        = 10;

  ChunkVectorTest chunktest;
  MockChunkVector chunkmock;

  for (int i = 0; i < size; i++) {
    chunktest.push_back(std::make_unique<MockChunk>(i));
    chunkmock.push_back(std::make_unique<MockChunk>(i));
  }

  SECTION("preconditioning")
  {
    REQUIRE(size == chunktest.size());
    REQUIRE(size == chunkmock.size());

    for (int i = 0; i < size; i++) {
      REQUIRE(chunktest[i]->get_id() == chunkmock[i]->get_id());
    }
  }

  SECTION("with pre-sorted data")
  {
    chunktest.sort_and_shrink();
    REQUIRE(chunktest.is_sorted());
  }

  SECTION("with shuffled data")
  {
    // random shuffle
    std::random_device seed_gen;
    std::mt19937       engine(seed_gen());
    std::shuffle(chunktest.begin(), chunktest.end(), engine);

    chunktest.sort_and_shrink();
    REQUIRE(chunktest.is_sorted());
  }

  SECTION("with nullptr")
  {
    // reset for even id
    for (int i = 0; i < chunktest.size(); i++) {
      if (chunktest[i]->get_id() % 2 == 0) {
        chunktest[i].reset();
      }
    }

    chunktest.sort_and_shrink();
    REQUIRE(chunktest.is_sorted());
    REQUIRE(chunktest.size() == size / 2);
    for (int i = 0; i < chunktest.size(); i++) {
      REQUIRE(chunktest[i]->get_id() % 2 != 0);
    }
  }
}

TEST_CASE("set_neighbors")
{
  int Cx = GENERATE(1, 4);
  int Cy = GENERATE(1, 4);
  int Cz = GENERATE(1, 4);

  std::unique_ptr<ChunkMap> chunkmap = std::make_unique<MockChunkMap>(Cz, Cy, Cx);
  ChunkVectorTest           chunktest;

  for (int i = 0; i < Cz * Cy * Cx; i++) {
    chunktest.push_back(std::make_unique<MockChunk>(i));
  }

  // just check if call does not crash
  chunktest.set_neighbors(chunkmap);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
