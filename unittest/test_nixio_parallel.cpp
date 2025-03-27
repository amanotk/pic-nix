// -*- C++ -*-

#include "nixio.hpp"
#include "xtensorall.hpp"
#include <fstream>
#include <iostream>

#include "catch.hpp"

using nixio::float32;
using nixio::float64;

// MPI option from command line
extern int options_mpi_decomposition[];

static const char          filename[] = "test_nixio_parallel_data.dat";
static const size_t        Nx         = 4;
static const size_t        Ny         = 4;
static const size_t        Nz         = 4;
static const size_t        N          = Nx * Ny * Nz;
static std::vector<size_t> gshape     = {Nz, Ny, Nx};

static int32_t i32a[N];
static int64_t i64a[N];
static float32 f32a[N];
static float64 f64a[N];
static auto    xt_i32a = xt::adapt(&i32a[0], N, xt::no_ownership(), gshape);
static auto    xt_i64a = xt::adapt(&i64a[0], N, xt::no_ownership(), gshape);
static auto    xt_f32a = xt::adapt(&f32a[0], N, xt::no_ownership(), gshape);
static auto    xt_f64a = xt::adapt(&f64a[0], N, xt::no_ownership(), gshape);

int get_thisrank()
{
  int thisrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
  return thisrank;
}

int get_nprocess()
{
  int nprocess;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  return nprocess;
}

bool parallel_decomposition(const size_t ndim, size_t shape[], size_t coord[])
{
  REQUIRE((ndim >= 1 && ndim <= 3));

  int thisrank = get_thisrank();
  int nprocess = get_nprocess();
  int procz    = options_mpi_decomposition[0];
  int procy    = options_mpi_decomposition[1];
  int procx    = options_mpi_decomposition[2];

  if (ndim == 1) {
    REQUIRE(nprocess == procx);
    REQUIRE(Nx % procx == 0);
    shape[0] = Nx / procx;
    coord[0] = thisrank;
  } else if (ndim == 2) {
    REQUIRE(nprocess == procx * procy);
    REQUIRE(Ny % procy == 0);
    REQUIRE(Nx % procx == 0);
    shape[0] = Ny / procy;
    shape[1] = Nx / procx;
    coord[0] = thisrank / procx;
    coord[1] = thisrank % procx;
  } else if (ndim == 3) {
    REQUIRE(nprocess == procx * procy * procz);
    REQUIRE(Nz % procz == 0);
    REQUIRE(Ny % procy == 0);
    REQUIRE(Nx % procx == 0);
    shape[0] = Nz / procz;
    shape[1] = Ny / procy;
    shape[2] = Nx / procx;
    coord[0] = thisrank / (procx * procy);
    coord[1] = (thisrank / procx) % procy;
    coord[2] = thisrank % procx;
  }

  return true;
}

template <typename T>
void init_array_ordered(const int N, T x[])
{
  for (int i = 0; i < N; i++) {
    x[i] = static_cast<T>(i);
  }
}

template <typename T>
void init_array_random(const int N, T x[])
{
  static std::random_device                rd;
  static std::mt19937                      mt(rd());
  static std::uniform_real_distribution<T> rand(0, N);

  for (int i = 0; i < N; i++) {
    x[i] = rand(mt);
  }
}

template <typename T>
bool is_array_equal(const int N, T x[], T y[])
{
  return std::equal(&x[0], &x[N - 1], y);
}

//
// read_single
//
TEST_CASE("ReadSingle")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  if (get_thisrank() == 0) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<char*>(i32a), sizeof(int32_t) * N);
    ofs.write(reinterpret_cast<char*>(i64a), sizeof(int64_t) * N);
    ofs.write(reinterpret_cast<char*>(f32a), sizeof(float32) * N);
    ofs.write(reinterpret_cast<char*>(f64a), sizeof(float64) * N);
    ofs.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;
    int32_t     in_i32a[N];
    int64_t     in_i64a[N];
    float32     in_f32a[N];
    float64     in_f64a[N];

    nixio::open_file(filename, &fh, &disp, "r");

    nixio::read_single(&fh, &disp, in_i32a, sizeof(int32_t) * N, &req[0]);
    nixio::read_single(&fh, &disp, in_i64a, sizeof(int64_t) * N, &req[1]);
    nixio::read_single(&fh, &disp, in_f32a, sizeof(float32) * N, &req[2]);
    nixio::read_single(&fh, &disp, in_f64a, sizeof(float64) * N, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    nixio::close_file(&fh);

    REQUIRE(is_array_equal(N, i32a, in_i32a));
    REQUIRE(is_array_equal(N, i64a, in_i64a));
    REQUIRE(is_array_equal(N, f32a, in_f32a));
    REQUIRE(is_array_equal(N, f64a, in_f64a));
  }

  // delete
  if (get_thisrank() == 0) {
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// write_single
//
TEST_CASE("WriteSingle")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;

    nixio::open_file(filename, &fh, &disp, "w");

    nixio::write_single(&fh, &disp, i32a, sizeof(int32_t) * N, &req[0]);
    nixio::write_single(&fh, &disp, i64a, sizeof(int64_t) * N, &req[1]);
    nixio::write_single(&fh, &disp, f32a, sizeof(float32) * N, &req[2]);
    nixio::write_single(&fh, &disp, f64a, sizeof(float64) * N, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    nixio::close_file(&fh);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  if (get_thisrank() == 0) {
    int32_t in_i32a[N];
    int64_t in_i64a[N];
    float32 in_f32a[N];
    float64 in_f64a[N];

    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(in_i32a), sizeof(int32_t) * N);
    ifs.read(reinterpret_cast<char*>(in_i64a), sizeof(int64_t) * N);
    ifs.read(reinterpret_cast<char*>(in_f32a), sizeof(float32) * N);
    ifs.read(reinterpret_cast<char*>(in_f64a), sizeof(float64) * N);
    ifs.close();

    REQUIRE(is_array_equal(N, i32a, in_i32a));
    REQUIRE(is_array_equal(N, i64a, in_i64a));
    REQUIRE(is_array_equal(N, f32a, in_f32a));
    REQUIRE(is_array_equal(N, f64a, in_f64a));

    // delete
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// read_contiguous
//
TEST_CASE("ReadContiguous")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  if (get_thisrank() == 0) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<char*>(i32a), sizeof(int32_t) * N);
    ofs.write(reinterpret_cast<char*>(i64a), sizeof(int64_t) * N);
    ofs.write(reinterpret_cast<char*>(f32a), sizeof(float32) * N);
    ofs.write(reinterpret_cast<char*>(f64a), sizeof(float64) * N);
    ofs.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;
    int32_t     in_i32a[N];
    int64_t     in_i64a[N];
    float32     in_f32a[N];
    float64     in_f64a[N];

    int              thisrank = get_thisrank();
    int              nprocess = get_nprocess();
    std::vector<int> boundary(nprocess + 1);

    for (int i = 0; i < nprocess + 1; i++) {
      boundary[i] = N / nprocess * i;
    }
    int p = boundary[thisrank];
    int s = boundary[thisrank + 1] - boundary[thisrank];

    nixio::open_file(filename, &fh, &disp, "r");

    nixio::read_contiguous(&fh, &disp, &in_i32a[p], s, sizeof(int32_t), 1, &req[0]);
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    nixio::read_contiguous(&fh, &disp, &in_i64a[p], s, sizeof(int64_t), 1, &req[1]);
    MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    nixio::read_contiguous(&fh, &disp, &in_f32a[p], s, sizeof(float32), 1, &req[2]);
    MPI_Wait(&req[2], MPI_STATUS_IGNORE);
    nixio::read_contiguous(&fh, &disp, &in_f64a[p], s, sizeof(float64), 1, &req[3]);
    MPI_Wait(&req[3], MPI_STATUS_IGNORE);

    nixio::close_file(&fh);

    REQUIRE(is_array_equal(s, &i32a[p], &in_i32a[p]));
    REQUIRE(is_array_equal(s, &i64a[p], &in_i64a[p]));
    REQUIRE(is_array_equal(s, &f32a[p], &in_f32a[p]));
    REQUIRE(is_array_equal(s, &f64a[p], &in_f64a[p]));
  }

  // delete
  if (get_thisrank() == 0) {
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// write_contiguous
//
TEST_CASE("WriteContiguous")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;

    int thisrank = get_thisrank();
    int nprocess = get_nprocess();

    std::vector<int> boundary(nprocess + 1);

    for (int i = 0; i < nprocess + 1; i++) {
      boundary[i] = N / nprocess * i;
    }
    int p = boundary[thisrank];
    int s = boundary[thisrank + 1] - boundary[thisrank];

    nixio::open_file(filename, &fh, &disp, "w");

    nixio::write_contiguous(&fh, &disp, &i32a[p], s, sizeof(int32_t), 1, &req[0]);
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    nixio::write_contiguous(&fh, &disp, &i64a[p], s, sizeof(int64_t), 1, &req[1]);
    MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    nixio::write_contiguous(&fh, &disp, &f32a[p], s, sizeof(float32), 1, &req[2]);
    MPI_Wait(&req[2], MPI_STATUS_IGNORE);
    nixio::write_contiguous(&fh, &disp, &f64a[p], s, sizeof(float64), 1, &req[3]);
    MPI_Wait(&req[3], MPI_STATUS_IGNORE);

    nixio::close_file(&fh);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  if (get_thisrank() == 0) {
    int32_t in_i32a[N];
    int64_t in_i64a[N];
    float32 in_f32a[N];
    float64 in_f64a[N];

    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(in_i32a), sizeof(int32_t) * N);
    ifs.read(reinterpret_cast<char*>(in_i64a), sizeof(int64_t) * N);
    ifs.read(reinterpret_cast<char*>(in_f32a), sizeof(float32) * N);
    ifs.read(reinterpret_cast<char*>(in_f64a), sizeof(float64) * N);
    ifs.close();

    REQUIRE(is_array_equal(N, i32a, in_i32a));
    REQUIRE(is_array_equal(N, i64a, in_i64a));
    REQUIRE(is_array_equal(N, f32a, in_f32a));
    REQUIRE(is_array_equal(N, f64a, in_f64a));

    // delete
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// read_contiguous_at
//
TEST_CASE("ReadContiguousAt")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  if (get_thisrank() == 0) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<char*>(i32a), sizeof(int32_t) * N);
    ofs.write(reinterpret_cast<char*>(i64a), sizeof(int64_t) * N);
    ofs.write(reinterpret_cast<char*>(f32a), sizeof(float32) * N);
    ofs.write(reinterpret_cast<char*>(f64a), sizeof(float64) * N);
    ofs.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp0, disp;
    int32_t     in_i32a[N];
    int64_t     in_i64a[N];
    float32     in_f32a[N];
    float64     in_f64a[N];

    int              thisrank = get_thisrank();
    int              nprocess = get_nprocess();
    std::vector<int> boundary(nprocess + 1);

    for (int i = 0; i < nprocess + 1; i++) {
      boundary[i] = N / nprocess * i;
    }
    int p = boundary[thisrank];
    int s = boundary[thisrank + 1] - boundary[thisrank];

    nixio::open_file(filename, &fh, &disp0, "r");

    disp = disp0 + p * sizeof(int32_t);
    nixio::read_contiguous_at(&fh, &disp, &in_i32a[p], s, sizeof(int32_t), &req[0]);

    disp0 = disp0 + N * sizeof(int32_t);
    disp  = disp0 + p * sizeof(int64_t);
    nixio::read_contiguous_at(&fh, &disp, &in_i64a[p], s, sizeof(int64_t), &req[1]);

    disp0 = disp0 + N * sizeof(int64_t);
    disp  = disp0 + p * sizeof(float32);
    nixio::read_contiguous_at(&fh, &disp, &in_f32a[p], s, sizeof(float32), &req[2]);

    disp0 = disp0 + N * sizeof(float32);
    disp  = disp0 + p * sizeof(float64);
    nixio::read_contiguous_at(&fh, &disp, &in_f64a[p], s, sizeof(float64), &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    nixio::close_file(&fh);

    REQUIRE(is_array_equal(s, &i32a[p], &in_i32a[p]));
    REQUIRE(is_array_equal(s, &i64a[p], &in_i64a[p]));
    REQUIRE(is_array_equal(s, &f32a[p], &in_f32a[p]));
    REQUIRE(is_array_equal(s, &f64a[p], &in_f64a[p]));
  }

  // delete
  if (get_thisrank() == 0) {
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// write_contiguous_at
//
TEST_CASE("WriteContiguousAt")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp0, disp;

    int thisrank = get_thisrank();
    int nprocess = get_nprocess();

    std::vector<int> boundary(nprocess + 1);

    for (int i = 0; i < nprocess + 1; i++) {
      boundary[i] = N / nprocess * i;
    }
    int p = boundary[thisrank];
    int s = boundary[thisrank + 1] - boundary[thisrank];

    nixio::open_file(filename, &fh, &disp0, "w");

    disp = disp0 + p * sizeof(int32_t);
    nixio::write_contiguous_at(&fh, &disp, &i32a[p], s, sizeof(int32_t), &req[0]);

    disp0 = disp0 + N * sizeof(int32_t);
    disp  = disp0 + p * sizeof(int64_t);
    nixio::write_contiguous_at(&fh, &disp, &i64a[p], s, sizeof(int64_t), &req[1]);

    disp0 = disp0 + N * sizeof(int64_t);
    disp  = disp0 + p * sizeof(float32);
    nixio::write_contiguous_at(&fh, &disp, &f32a[p], s, sizeof(float32), &req[2]);

    disp0 = disp0 + N * sizeof(float32);
    disp  = disp0 + p * sizeof(float64);
    nixio::write_contiguous_at(&fh, &disp, &f64a[p], s, sizeof(float64), &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    nixio::close_file(&fh);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  if (get_thisrank() == 0) {
    int32_t in_i32a[N];
    int64_t in_i64a[N];
    float32 in_f32a[N];
    float64 in_f64a[N];

    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(in_i32a), sizeof(int32_t) * N);
    ifs.read(reinterpret_cast<char*>(in_i64a), sizeof(int64_t) * N);
    ifs.read(reinterpret_cast<char*>(in_f32a), sizeof(float32) * N);
    ifs.read(reinterpret_cast<char*>(in_f64a), sizeof(float64) * N);
    ifs.close();

    REQUIRE(is_array_equal(N, i32a, in_i32a));
    REQUIRE(is_array_equal(N, i64a, in_i64a));
    REQUIRE(is_array_equal(N, f32a, in_f32a));
    REQUIRE(is_array_equal(N, f64a, in_f64a));

    // delete
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// read_subarray
//
TEST_CASE("ReadSubarray")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  // decomposition
  const int           ndim = 3;
  std::vector<size_t> lshape(ndim);
  std::vector<size_t> offset(ndim);
  std::vector<size_t> coord(ndim);

  parallel_decomposition(ndim, lshape.data(), coord.data());
  offset[0] = lshape[0] * coord[0];
  offset[1] = lshape[1] * coord[1];
  offset[2] = lshape[2] * coord[2];

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  if (get_thisrank() == 0) {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs.write(reinterpret_cast<char*>(i32a), sizeof(int32_t) * N);
    ofs.write(reinterpret_cast<char*>(i64a), sizeof(int64_t) * N);
    ofs.write(reinterpret_cast<char*>(f32a), sizeof(float32) * N);
    ofs.write(reinterpret_cast<char*>(f64a), sizeof(float64) * N);
    ofs.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;
    size_t      nd = ndim;
    size_t*     gs = gshape.data();
    size_t*     ls = lshape.data();
    size_t*     os = offset.data();

    // local array
    xt::xarray<int32_t> in_i32a(lshape);
    xt::xarray<int64_t> in_i64a(lshape);
    xt::xarray<float32> in_f32a(lshape);
    xt::xarray<float64> in_f64a(lshape);

    nixio::open_file(filename, &fh, &disp, "r");

    nixio::read_subarray(&fh, &disp, in_i32a.data(), nd, gs, ls, os, sizeof(int32_t), &req[0]);
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    nixio::read_subarray(&fh, &disp, in_i64a.data(), nd, gs, ls, os, sizeof(int64_t), &req[1]);
    MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    nixio::read_subarray(&fh, &disp, in_f32a.data(), nd, gs, ls, os, sizeof(float32), &req[2]);
    MPI_Wait(&req[2], MPI_STATUS_IGNORE);
    nixio::read_subarray(&fh, &disp, in_f64a.data(), nd, gs, ls, os, sizeof(float64), &req[3]);
    MPI_Wait(&req[3], MPI_STATUS_IGNORE);

    nixio::close_file(&fh);

    xt::xarray<int32_t> local_i32a(lshape);
    xt::xarray<int64_t> local_i64a(lshape);
    xt::xarray<float32> local_f32a(lshape);
    xt::xarray<float64> local_f64a(lshape);
    auto                zrange = xt::range(offset[0], offset[0] + lshape[0]);
    auto                yrange = xt::range(offset[1], offset[1] + lshape[1]);
    auto                xrange = xt::range(offset[2], offset[2] + lshape[2]);
    local_i32a                 = xt::strided_view(xt_i32a, {zrange, yrange, xrange});
    local_i64a                 = xt::strided_view(xt_i64a, {zrange, yrange, xrange});
    local_f32a                 = xt::strided_view(xt_f32a, {zrange, yrange, xrange});
    local_f64a                 = xt::strided_view(xt_f64a, {zrange, yrange, xrange});

    int count = N / get_nprocess();
    REQUIRE(is_array_equal(count, local_i32a.data(), in_i32a.data()));
    REQUIRE(is_array_equal(count, local_i64a.data(), in_i64a.data()));
    REQUIRE(is_array_equal(count, local_f32a.data(), in_f32a.data()));
    REQUIRE(is_array_equal(count, local_f64a.data(), in_f64a.data()));
  }

  // delete
  if (get_thisrank() == 0) {
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//
// write_subarray
//
TEST_CASE("WriteSubarray")
{
  init_array_ordered(N, i32a);
  init_array_ordered(N, i64a);
  init_array_ordered(N, f32a);
  init_array_ordered(N, f64a);

  // decomposition
  const int           ndim = 3;
  std::vector<size_t> lshape(ndim);
  std::vector<size_t> offset(ndim);
  std::vector<size_t> coord(ndim);

  parallel_decomposition(ndim, lshape.data(), coord.data());
  offset[0] = lshape[0] * coord[0];
  offset[1] = lshape[1] * coord[1];
  offset[2] = lshape[2] * coord[2];

  MPI_Barrier(MPI_COMM_WORLD);

  // write to file
  {
    MPI_File    fh;
    MPI_Request req[4];
    size_t      disp;
    size_t      nd = ndim;
    size_t*     gs = gshape.data();
    size_t*     ls = lshape.data();
    size_t*     os = offset.data();

    // local array
    xt::xarray<int32_t> local_i32a(lshape);
    xt::xarray<int64_t> local_i64a(lshape);
    xt::xarray<float32> local_f32a(lshape);
    xt::xarray<float64> local_f64a(lshape);
    auto                zrange = xt::range(offset[0], offset[0] + lshape[0]);
    auto                yrange = xt::range(offset[1], offset[1] + lshape[1]);
    auto                xrange = xt::range(offset[2], offset[2] + lshape[2]);
    local_i32a                 = xt::strided_view(xt_i32a, {zrange, yrange, xrange});
    local_i64a                 = xt::strided_view(xt_i64a, {zrange, yrange, xrange});
    local_f32a                 = xt::strided_view(xt_f32a, {zrange, yrange, xrange});
    local_f64a                 = xt::strided_view(xt_f64a, {zrange, yrange, xrange});

    nixio::open_file(filename, &fh, &disp, "w");

    nixio::write_subarray(&fh, &disp, local_i32a.data(), nd, gs, ls, os, sizeof(int32_t), &req[0]);
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
    nixio::write_subarray(&fh, &disp, local_i64a.data(), nd, gs, ls, os, sizeof(int64_t), &req[1]);
    MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    nixio::write_subarray(&fh, &disp, local_f32a.data(), nd, gs, ls, os, sizeof(float32), &req[2]);
    MPI_Wait(&req[2], MPI_STATUS_IGNORE);
    nixio::write_subarray(&fh, &disp, local_f64a.data(), nd, gs, ls, os, sizeof(float64), &req[3]);
    MPI_Wait(&req[3], MPI_STATUS_IGNORE);

    nixio::close_file(&fh);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // read and check results
  if (get_thisrank() == 0) {
    int32_t in_i32a[N];
    int64_t in_i64a[N];
    float32 in_f32a[N];
    float64 in_f64a[N];

    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    ifs.read(reinterpret_cast<char*>(in_i32a), sizeof(int32_t) * N);
    ifs.read(reinterpret_cast<char*>(in_i64a), sizeof(int64_t) * N);
    ifs.read(reinterpret_cast<char*>(in_f32a), sizeof(float32) * N);
    ifs.read(reinterpret_cast<char*>(in_f64a), sizeof(float64) * N);
    ifs.close();

    REQUIRE(is_array_equal(N, xt_i32a.data(), in_i32a));
    REQUIRE(is_array_equal(N, xt_i64a.data(), in_i64a));
    REQUIRE(is_array_equal(N, xt_f32a.data(), in_f32a));
    REQUIRE(is_array_equal(N, xt_f64a.data(), in_f64a));

    // delete
    std::remove(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
