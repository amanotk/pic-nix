// -*- C++ -*-

#include "chunk.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix;

//
// dummy for testing
//
class TestChunk : public Chunk
{
public:
  static const int* defaultDims()
  {
    static const int dims[3] = {1, 1, 1};
    return dims;
  }
  static const bool* defaultHasDim()
  {
    static const bool has_dim[3] = {true, true, true};
    return has_dim;
  }

  TestChunk() : Chunk(defaultDims(), defaultHasDim(), 0)
  {
  }

  virtual void setup(json& config) override
  {
  }
};

//
// Chunk
//
TEST_CASE("Chunk")
{
  const int bufsize = 4096;
  const int N       = 5;

  char      buffer[bufsize];
  int       chunkid[N][N][N];
  TestChunk chunk[N][N][N];
  TestChunk unpack_chunk;

  // set IDs
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        chunkid[i][j][k] = N * N * i + N * j + k;
        chunk[i][j][k].set_id(chunkid[i][j][k]);
      }
    }
  }

  // set neighbor IDs
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      for (int k = 1; k < N - 1; k++) {
        //
        chunk[i][j][k].set_nb_id(-1, -1, -1, chunk[i - 1][j - 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(-1, -1, +0, chunk[i - 1][j - 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(-1, -1, +1, chunk[i - 1][j - 1][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(-1, +0, -1, chunk[i - 1][j + 0][k - 1].get_id());
        chunk[i][j][k].set_nb_id(-1, +0, +0, chunk[i - 1][j + 0][k + 0].get_id());
        chunk[i][j][k].set_nb_id(-1, +0, +1, chunk[i - 1][j + 0][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(-1, +1, -1, chunk[i - 1][j + 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(-1, +1, +0, chunk[i - 1][j + 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(-1, +1, +1, chunk[i - 1][j + 1][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+0, -1, -1, chunk[i + 0][j - 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+0, -1, +0, chunk[i + 0][j - 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+0, -1, +1, chunk[i + 0][j - 1][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+0, +0, -1, chunk[i + 0][j + 0][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+0, +0, +0, chunk[i + 0][j + 0][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+0, +0, +1, chunk[i + 0][j + 0][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+0, +1, -1, chunk[i + 0][j + 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+0, +1, +0, chunk[i + 0][j + 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+0, +1, +1, chunk[i + 0][j + 1][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+1, -1, -1, chunk[i + 1][j - 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+1, -1, +0, chunk[i + 1][j - 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+1, -1, +1, chunk[i + 1][j - 1][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+1, +0, -1, chunk[i + 1][j + 0][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+1, +0, +0, chunk[i + 1][j + 0][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+1, +0, +1, chunk[i + 1][j + 0][k + 1].get_id());
        //
        chunk[i][j][k].set_nb_id(+1, +1, -1, chunk[i + 1][j + 1][k - 1].get_id());
        chunk[i][j][k].set_nb_id(+1, +1, +0, chunk[i + 1][j + 1][k + 0].get_id());
        chunk[i][j][k].set_nb_id(+1, +1, +1, chunk[i + 1][j + 1][k + 1].get_id());
      }
    }
  }

  //
  // check send/recv tag
  //
  INFO("send/recv tags");
  {
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        for (int k = -1; k <= +1; k++) {
          int is = 2, ir = 2 + i;
          int js = 2, jr = 2 + j;
          int ks = 2, kr = 2 + k;
          REQUIRE(chunk[is][js][ks].get_sndtag(+i, +j, +k) ==
                  chunk[ir][jr][kr].get_rcvtag(-i, -j, -k));
        }
      }
    }
    // test recv tags
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        for (int k = -1; k <= +1; k++) {
          int ir = 2, is = 2 + i;
          int jr = 2, js = 2 + j;
          int kr = 2, ks = 2 + k;
          REQUIRE(chunk[ir][jr][kr].get_rcvtag(+i, +j, +k) ==
                  chunk[is][js][ks].get_sndtag(-i, -j, -k));
        }
      }
    }
  }

  //
  // check pack/unpack
  //
  INFO("pack/unpack");
  {
    int byte1 = chunk[2][2][2].pack(buffer, 0);
    int byte2 = unpack_chunk.unpack(buffer, 0);

    REQUIRE(byte1 == byte2);
    REQUIRE(chunk[2][2][2].get_id() == unpack_chunk.get_id());
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        for (int k = -1; k <= +1; k++) {
          int ir = 2, is = 2 + i;
          int jr = 2, js = 2 + j;
          int kr = 2, ks = 2 + k;
          REQUIRE(chunk[2][2][2].get_rcvtag(+i, +j, +k) == unpack_chunk.get_rcvtag(+i, +j, +k));
          REQUIRE(chunk[2][2][2].get_sndtag(+i, +j, +k) == unpack_chunk.get_sndtag(+i, +j, +k));
        }
      }
    }
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
