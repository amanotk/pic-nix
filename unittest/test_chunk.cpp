// -*- C++ -*-

#include "chunk.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix;

//
// dummy for testing
//
template <int Nb>
class TestChunk : public Chunk<Nb>
{
  virtual int set_boundary_query(int mode, int sendrecv) override
  {
    return 0;
  }

  virtual void set_boundary_pack(int mode) override
  {
  }

  virtual void set_boundary_unpack(int mode) override
  {
  }

  virtual void set_boundary_begin(int mode) override
  {
  }

  virtual void set_boundary_end(int mode) override
  {
  }
};

//
// Chunk<1>
//
TEST_CASE("Chunk1")
{
  const int bufsize = 4096;
  const int N       = 5;

  char         buffer[bufsize];
  int          chunkid[N];
  TestChunk<1> chunk[N];
  TestChunk<1> unpack_chunk;

  // set IDs
  for (int i = 0; i < N; i++) {
    chunkid[i] = i;
    chunk[i].set_id(chunkid[i]);
  }

  // set neighbor IDs
  for (int i = 1; i < N - 1; i++) {
    chunk[i].set_nb_id(-1, chunk[i - 1].get_id());
    chunk[i].set_nb_id(+0, chunk[i + 0].get_id());
    chunk[i].set_nb_id(+1, chunk[i + 1].get_id());
  }

  //
  // check send/recv tag
  //
  INFO("send/recv tags");
  {
    for (int i = -1; i <= +1; i++) {
      int is = 2, ir = 2 + i;
      REQUIRE(chunk[is].get_sndtag(+i) == chunk[ir].get_rcvtag(-i));
    }
    // test recv tags
    for (int i = -1; i <= +1; i++) {
      int ir = 2, is = 2 + i;
      REQUIRE(chunk[ir].get_rcvtag(+i) == chunk[is].get_sndtag(-i));
    }
  }

  //
  // check pack/unpack
  //
  INFO("pack/unpack");
  {
    int byte1 = chunk[2].pack(buffer, 0);
    int byte2 = unpack_chunk.unpack(buffer, 0);

    REQUIRE(byte1 == byte2);
    REQUIRE(chunk[2].get_id() == unpack_chunk.get_id());
    for (int i = -1; i <= +1; i++) {
      int ir = 2, is = 2 + i;
      REQUIRE(chunk[2].get_rcvtag(+i) == unpack_chunk.get_rcvtag(+i));
      REQUIRE(chunk[2].get_sndtag(+i) == unpack_chunk.get_sndtag(+i));
    }
  }
}

//
// Chunk<2>
//
TEST_CASE("Chunk2")
{
  const int bufsize = 4096;
  const int N       = 5;

  char         buffer[bufsize];
  int          chunkid[N][N];
  TestChunk<2> chunk[N][N];
  TestChunk<2> unpack_chunk;

  // set IDs
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      chunkid[i][j] = N * i + j;
      chunk[i][j].set_id(chunkid[i][j]);
    }
  }

  // set neighbor IDs
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      //
      chunk[i][j].set_nb_id(-1, -1, chunk[i - 1][j - 1].get_id());
      chunk[i][j].set_nb_id(-1, +0, chunk[i - 1][j + 0].get_id());
      chunk[i][j].set_nb_id(-1, +1, chunk[i - 1][j + 1].get_id());
      //
      chunk[i][j].set_nb_id(+0, -1, chunk[i + 0][j - 1].get_id());
      chunk[i][j].set_nb_id(+0, +0, chunk[i + 0][j + 0].get_id());
      chunk[i][j].set_nb_id(+0, +1, chunk[i + 0][j + 1].get_id());
      //
      chunk[i][j].set_nb_id(+1, -1, chunk[i + 1][j - 1].get_id());
      chunk[i][j].set_nb_id(+1, +0, chunk[i + 1][j + 0].get_id());
      chunk[i][j].set_nb_id(+1, +1, chunk[i + 1][j + 1].get_id());
    }
  }

  //
  // check send/recv tag
  //
  INFO("send/recv tags");
  {
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        int is = 2, ir = 2 + i;
        int js = 2, jr = 2 + j;
        REQUIRE(chunk[is][js].get_sndtag(+i, +j) == chunk[ir][jr].get_rcvtag(-i, -j));
      }
    }
    // test recv tags
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        int ir = 2, is = 2 + i;
        int jr = 2, js = 2 + j;
        REQUIRE(chunk[ir][jr].get_rcvtag(+i, +j) == chunk[is][js].get_sndtag(-i, -j));
      }
    }
  }

  //
  // check pack/unpack
  //
  INFO("pack/unpack");
  {
    int byte1 = chunk[2][2].pack(buffer, 0);
    int byte2 = unpack_chunk.unpack(buffer, 0);

    REQUIRE(byte1 == byte2);
    REQUIRE(chunk[2][2].get_id() == unpack_chunk.get_id());
    for (int i = -1; i <= +1; i++) {
      for (int j = -1; j <= +1; j++) {
        int ir = 2, is = 2 + i;
        int jr = 2, js = 2 + j;
        REQUIRE(chunk[2][2].get_rcvtag(+i, +j) == unpack_chunk.get_rcvtag(+i, +j));
        REQUIRE(chunk[2][2].get_sndtag(+i, +j) == unpack_chunk.get_sndtag(+i, +j));
      }
    }
  }
}

//
// Chunk<3>
//
TEST_CASE("Chunk3")
{
  const int bufsize = 4096;
  const int N       = 5;

  char         buffer[bufsize];
  int          chunkid[N][N][N];
  TestChunk<3> chunk[N][N][N];
  TestChunk<3> unpack_chunk;

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
