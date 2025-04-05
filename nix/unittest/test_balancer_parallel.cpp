// -*- C++ -*-

#include "balancer.hpp"
#include "chunkvector.hpp"

#include <iostream>

#include "catch.hpp"

using namespace nix;

class MockChunk;
class MockChunkMap;
class MockApplicationInterface;
using ChunkVec     = ChunkVector<std::unique_ptr<MockChunk>>;
using PtrChunkMap  = std::unique_ptr<MockChunkMap>;
using PtrInterface = std::shared_ptr<MockApplicationInterface>;

static const int ndata    = 10;
static int       ndims[4] = {8, 8, 8, 8 * 8 * 8};
static int       cdims[4] = {2, 2, 2, 2 * 2 * 2};

static int find_index(std::vector<int>& boundary, int id)
{
  auto it = std::upper_bound(boundary.begin(), boundary.end(), id);
  return std::distance(boundary.begin(), it) - 1;
}

static float64 get_x_value(int id, int index)
{
  return 100 * id + index;
}

class MockChunk
{
private:
  int                  myid;
  float64              myload;
  std::vector<float64> x;

public:
  MockChunk(const int dims[3], const bool has_dim[3], int id) : myid(id)
  {
    x.resize(ndata);
  }

  void set_load(float64 load)
  {
    myload = load;
  }

  void set_id(int id)
  {
    myid = id;
  }

  int get_id()
  {
    return myid;
  }

  void set_data(std::vector<float64>& xx)
  {
    x = xx;
  }

  std::vector<float64>& get_data()
  {
    return x;
  }

  float64 get_total_load()
  {
    return myload;
  }

  int pack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(buffer, &myid, sizeof(int), count, 0);
    count += memcpy_count(buffer, x.data(), sizeof(float64) * ndata, count, 0);

    return count;
  }

  int unpack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(&myid, buffer, sizeof(int), 0, count);
    count += memcpy_count(x.data(), buffer, sizeof(float64) * ndata, 0, count);

    return count;
  }
};

class MockChunkMap
{
private:
  std::vector<int> boundary;

public:
  MockChunkMap(std::vector<int>& boundary) : boundary(boundary)
  {
  }

  int get_rank(int id)
  {
    return find_index(boundary, id);
  }
};

class MockApplicationInterface
{
public:
  using PtrChunkMap = std::unique_ptr<MockChunkMap>;
  using PtrChunk    = std::unique_ptr<MockChunk>;
  using ChunkVec    = ChunkVector<PtrChunk>;

  struct DataContainer {
    int*         ndims;
    int*         cdims;
    int&         thisrank;
    int&         nprocess;
    PtrChunkMap& chunkmap;
    ChunkVec&    chunkvec;
  };

  DataContainer data;

  MockApplicationInterface(int* ndims, int* cdims, int& thisrank, int& nprocess,
                           PtrChunkMap& chunkmap, ChunkVec& chunkvec)
      : data{ndims, cdims, thisrank, nprocess, chunkmap, chunkvec}
  {
  }

  ~MockApplicationInterface() = default;

  virtual DataContainer get_data()
  {
    return data;
  }

  virtual PtrChunk create_chunk(const int dims[3], const bool has_dim[3], int id)
  {
    return std::make_unique<MockChunk>(dims, has_dim, id);
  }
};

class TestBalancer : public Balancer
{
private:
  int nprocess;
  int thisrank;

public:
  TestBalancer(int nchunk) : Balancer(nchunk)
  {
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  }

  ~TestBalancer()
  {
  }

  ChunkVec create_chunkvec(std::vector<int>& boundary)
  {
    ChunkVec chunkvec;

    bool has_dim[3] = {true, true, true};

    for (int i = boundary[thisrank], j = 0; i < boundary[thisrank + 1]; i++, j++) {
      chunkvec.push_back(std::make_unique<MockChunk>(ndims, has_dim, i));
      chunkvec[j]->set_load(1.0 / (boundary[thisrank + 1] - boundary[thisrank]));
    }

    return chunkvec;
  }

  bool is_load_valid(std::vector<int>& boundary)
  {
    bool status = true;

    for (int i = 0; i < nchunk; i++) {
      int     index     = find_index(boundary, i);
      float64 true_load = 1.0 / (boundary[index + 1] - boundary[index]);

      status = status & (load(i) == true_load);
    }

    return status;
  }

  void set_chunkvec_data(ChunkVec& chunkvec)
  {
    for (int i = 0; i < chunkvec.size(); i++) {
      std::vector<float64> x(ndata);
      for (int j = 0; j < ndata; j++) {
        int id = chunkvec[i]->get_id();
        x[j]   = get_x_value(id, j);
      }
      chunkvec[i]->set_data(x);
    }
  }

  bool is_chunkvec_valid(ChunkVec& chunkvec, std::vector<int>& boundary)
  {
    bool status = true;

    status = status & (chunkvec.size() == boundary[thisrank + 1] - boundary[thisrank]);

    for (int i = 0, id = boundary[thisrank]; i < chunkvec.size(); i++, id++) {
      status = status & (chunkvec[i]->get_id() == id);
      for (int j = 0; j < ndata; j++) {
        status = status & (chunkvec[i]->get_data()[j] == get_x_value(id, j));
      }
    }

    return status;
  }

  void test_upload_global_load(std::vector<int>& boundary)
  {
    REQUIRE(nprocess == boundary.size() - 1);
    REQUIRE(boundary[0] == 0);
    REQUIRE(boundary[nprocess] == nchunk);

    PtrChunkMap  chunkmap  = std::make_unique<MockChunkMap>(boundary);
    ChunkVec     chunkvec  = create_chunkvec(boundary);
    PtrInterface interface = std::make_shared<MockApplicationInterface>(
        ndims, cdims, thisrank, nprocess, chunkmap, chunkvec);

    update_global_load(interface);

    REQUIRE(is_load_valid(boundary) == true);
  }

  void test_sendrecv_chunk(std::vector<int>& boundary)
  {
    REQUIRE(nprocess == boundary.size() - 1);
    REQUIRE(boundary[0] == 0);
    REQUIRE(boundary[nprocess] == nchunk);

    // boundary before sendrecv
    std::vector<int> initial_boundary = {0, 4, 8, 12, 16, 20, 24, 28, 32};

    PtrChunkMap     chunkmap  = std::make_unique<MockChunkMap>(initial_boundary);
    ChunkVec        chunkvec  = create_chunkvec(initial_boundary);
    PtrInterface    interface = std::make_shared<MockApplicationInterface>(
        ndims, cdims, thisrank, nprocess, chunkmap, chunkvec);

    set_chunkvec_data(chunkvec);
    sendrecv_chunk(interface, boundary);

    REQUIRE(is_chunkvec_valid(chunkvec, boundary) == true);
  }
};

TEST_CASE("test_uppdate_global_load")
{
  const int        nchunk   = 32;
  std::vector<int> boundary = {0, 5, 9, 12, 15, 19, 25, 27, 32};

  TestBalancer balancer(nchunk);

  balancer.fill_load(0.0);
  balancer.test_upload_global_load(boundary);
}

TEST_CASE("test_sendrecv_chunk")
{
  const int        nchunk   = 32;
  std::vector<int> boundary = {0, 5, 9, 12, 15, 19, 25, 27, 32};

  TestBalancer balancer(nchunk);

  balancer.test_sendrecv_chunk(boundary);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
