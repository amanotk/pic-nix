// -*- C++ -*-

#include "pic_chunk.hpp"
#include "test_parallel.hpp"

#include "nix/chunkmap.hpp"
#include "nix/chunkvector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <vector>

namespace
{
struct ExchangeContext {
  std::shared_ptr<PicChunk> chunk;
  nix::Dims3D               gdims;
  nix::Bool3D               has_dim;
  nix::Dims3D               local_dims;
  std::array<int, 3>        cdims;
  std::array<int, 3>        coord;
  std::array<int, 3>        offset;
  int                       boundary_margin;
};

struct TestGrid {
  nix::Dims3D        gdims;
  nix::Bool3D        has_dim;
  std::array<int, 3> cdims;
  int                mpi_size;
};

json make_boundary_config(int order)
{
  json config;
  config["option"]                  = json::object();
  config["option"]["vectorization"] = "scalar";
  config["option"]["order"]         = order;
  config["option"]["pusher"]        = "Boris";
  config["option"]["interpolation"] = "MC";
  config["option"]["seed_type"]     = "fixed";
  config["option"]["friedman"]      = 0.0;
  config["option"]["cell_load"]     = 1.0;
  config["option"]["buffer_ratio"]  = 0.2;
  return config;
}

void setup_mpi_buffers(PicChunk& chunk)
{
  chunk.set_mpi_buffer(chunk.get_mpi_buffer(BoundaryEmf), 0, 0, sizeof(float64) * 6);
  chunk.set_mpi_buffer(chunk.get_mpi_buffer(BoundaryCur), 0, 0, sizeof(float64) * 4);
  chunk.set_mpi_buffer(chunk.get_mpi_buffer(BoundaryParticle), 0,
                       nix::XtensorHaloParticle3D<PicChunk>::head_byte,
                       nix::XtensorHaloParticle3D<PicChunk>::elem_byte);
}

void setup_mpi_communicators(PicChunk& chunk)
{
  static std::vector<MPI_Comm> comms;
  if (comms.empty()) {
    comms.resize(static_cast<size_t>(NumBoundaryMode) * 27);
    for (int mode = 0; mode < NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            const int index = mode * 27 + iz * 9 + iy * 3 + ix;
            MPI_Comm_dup(MPI_COMM_WORLD, &comms[static_cast<size_t>(index)]);
          }
        }
      }
    }
  }

  for (int mode : {BoundaryEmf, BoundaryCur, BoundaryParticle}) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          chunk.set_mpi_communicator(mode, iz, iy, ix, comms[static_cast<size_t>(index)]);
        }
      }
    }
  }
}

ExchangeContext build_exchange_context(const std::array<int, 3>& cdims, const nix::Dims3D& gdims,
                                       const nix::Bool3D& has_dim, int order)
{
  const int nproc = get_mpi_size();
  const int rank  = get_mpi_rank();
  const int total = cdims[0] * cdims[1] * cdims[2];
  REQUIRE(nproc == total);

  nix::Dims3D dims = {gdims[0] / cdims[0], gdims[1] / cdims[1], gdims[2] / cdims[2]};

  auto             chunkmap = std::make_unique<nix::ChunkMap>(cdims[0], cdims[1], cdims[2]);
  std::vector<int> boundary(nproc + 1);
  for (int i = 0; i <= nproc; i++) {
    boundary[i] = i;
  }
  chunkmap->set_rank_boundary(boundary);

  int                                         id    = boundary[rank];
  auto                                        chunk = std::make_shared<PicChunk>(dims, has_dim, id);
  nix::ChunkVector<std::shared_ptr<PicChunk>> chunkvec;
  chunkvec.push_back(chunk);
  chunkvec.set_neighbors(chunkmap);

  auto [cz, cy, cx]         = chunkmap->get_coordinate(id);
  std::array<int, 3> coord  = {cz, cy, cx};
  std::array<int, 3> offset = {cz * dims[0], cy * dims[1], cx * dims[2]};
  chunk->set_global_context(offset.data(), gdims.data());
  chunk->set_coordinate(1.0, 1.0, 1.0);
  auto config = make_boundary_config(order);
  chunk->setup(config);
  chunk->allocate();
  setup_mpi_communicators(*chunk);
  setup_mpi_buffers(*chunk);

  return {chunk, gdims, has_dim, dims, cdims, coord, offset, chunk->get_boundary_margin()};
}

int wrap_index(int value, int size)
{
  int wrapped = value % size;
  if (wrapped < 0) {
    wrapped += size;
  }
  return wrapped;
}

int global_index(int local, int lb, int offset, int size)
{
  return wrap_index(offset + (local - lb), size);
}

int neighbor_coord(int coord, int dir, int dim, const std::array<int, 3>& cdims)
{
  int cdir = coord + dir;
  if (cdir < 0) {
    cdir = cdims[dim] - 1;
  } else if (cdir >= cdims[dim]) {
    cdir = 0;
  }
  return cdir;
}

int direction_for_index(int index, int lb, int ub)
{
  if (index < lb) {
    return -1;
  }
  if (index > ub) {
    return +1;
  }
  return 0;
}

int source_index(int index, int lb, int ub, int margin, int dir)
{
  if (dir == -1) {
    return ub - margin + 1 + (index - (lb - margin));
  }
  if (dir == 1) {
    return lb + (index - (ub + 1));
  }
  return index;
}

std::vector<int> current_direction_options(int index, int lb, int ub, int margin, bool has_dim)
{
  if (!has_dim) {
    return {0};
  }

  std::vector<int> dirs = {0};
  if (index >= lb && index <= lb + margin - 1) {
    dirs.push_back(-1);
  }
  if (index >= ub - margin + 1 && index <= ub) {
    dirs.push_back(1);
  }
  return dirs;
}

int source_index_current(int index, int lb, int ub, int margin, int dir)
{
  if (dir == -1) {
    return ub + 1 + (index - lb);
  }
  if (dir == 1) {
    return lb - margin + (index - (ub - margin + 1));
  }
  return index;
}

struct ExpectedValues {
  const ExchangeContext&         ctx;
  const PicChunk::DataContainer& data;
  float64                        base;

  float64 direct(int iz, int iy, int ix, int comp, const std::array<int, 3>& offset) const
  {
    const int gz = ctx.has_dim[0] ? global_index(iz, data.Lbz, offset[0], ctx.gdims[0]) : 0;
    const int gy = ctx.has_dim[1] ? global_index(iy, data.Lby, offset[1], ctx.gdims[1]) : 0;
    const int gx = ctx.has_dim[2] ? global_index(ix, data.Lbx, offset[2], ctx.gdims[2]) : 0;

    return base + 100.0 * gz + 10.0 * gy + static_cast<float64>(gx) + 0.001 * comp;
  }

  float64 field(int iz, int iy, int ix, int comp) const
  {
    const int dirz = ctx.has_dim[0] ? direction_for_index(iz, data.Lbz, data.Ubz) : 0;
    const int diry = ctx.has_dim[1] ? direction_for_index(iy, data.Lby, data.Uby) : 0;
    const int dirx = ctx.has_dim[2] ? direction_for_index(ix, data.Lbx, data.Ubx) : 0;

    const int src_cz = neighbor_coord(ctx.coord[0], dirz, 0, ctx.cdims);
    const int src_cy = neighbor_coord(ctx.coord[1], diry, 1, ctx.cdims);
    const int src_cx = neighbor_coord(ctx.coord[2], dirx, 2, ctx.cdims);

    const int src_iz =
        ctx.has_dim[0] ? source_index(iz, data.Lbz, data.Ubz, ctx.boundary_margin, dirz) : data.Lbz;
    const int src_iy =
        ctx.has_dim[1] ? source_index(iy, data.Lby, data.Uby, ctx.boundary_margin, diry) : data.Lby;
    const int src_ix =
        ctx.has_dim[2] ? source_index(ix, data.Lbx, data.Ubx, ctx.boundary_margin, dirx) : data.Lbx;

    const std::array<int, 3> offset = {
        src_cz * ctx.local_dims[0],
        src_cy * ctx.local_dims[1],
        src_cx * ctx.local_dims[2],
    };

    return direct(src_iz, src_iy, src_ix, comp, offset);
  }

  float64 current(int iz, int iy, int ix, int comp) const
  {
    float64 expected = direct(iz, iy, ix, comp, ctx.offset);

    if ((ctx.has_dim[0] && (iz < data.Lbz || iz > data.Ubz)) ||
        (ctx.has_dim[1] && (iy < data.Lby || iy > data.Uby)) ||
        (ctx.has_dim[2] && (ix < data.Lbx || ix > data.Ubx))) {
      return expected;
    }

    const auto dirz_options =
        current_direction_options(iz, data.Lbz, data.Ubz, ctx.boundary_margin, ctx.has_dim[0]);
    const auto diry_options =
        current_direction_options(iy, data.Lby, data.Uby, ctx.boundary_margin, ctx.has_dim[1]);
    const auto dirx_options =
        current_direction_options(ix, data.Lbx, data.Ubx, ctx.boundary_margin, ctx.has_dim[2]);

    for (int dirz : dirz_options) {
      for (int diry : diry_options) {
        for (int dirx : dirx_options) {
          if (dirz == 0 && diry == 0 && dirx == 0) {
            continue;
          }

          const int src_cz = neighbor_coord(ctx.coord[0], dirz, 0, ctx.cdims);
          const int src_cy = neighbor_coord(ctx.coord[1], diry, 1, ctx.cdims);
          const int src_cx = neighbor_coord(ctx.coord[2], dirx, 2, ctx.cdims);

          const int src_iz =
              source_index_current(iz, data.Lbz, data.Ubz, ctx.boundary_margin, dirz);
          const int src_iy =
              source_index_current(iy, data.Lby, data.Uby, ctx.boundary_margin, diry);
          const int src_ix =
              source_index_current(ix, data.Lbx, data.Ubx, ctx.boundary_margin, dirx);

          const std::array<int, 3> offset = {
              src_cz * ctx.local_dims[0],
              src_cy * ctx.local_dims[1],
              src_cx * ctx.local_dims[2],
          };

          expected += direct(src_iz, src_iy, src_ix, comp, offset);
        }
      }
    }

    return expected;
  }
};

void exchange_boundary(PicChunk& chunk, int mode)
{
  chunk.set_boundary_pack(mode);
  chunk.set_boundary_begin(mode);
  if (mode == BoundaryParticle) {
    chunk.set_boundary_probe(mode, true);
  }
  chunk.set_boundary_end(mode);
  chunk.set_boundary_unpack(mode);
}

template <typename Array>
void fill_interior(Array& array, const ExchangeContext& ctx, const PicChunk::DataContainer& data,
                   float64 base, int ncomp)
{
  ExpectedValues expected{ctx, data, base};
  array.fill(-1.0);
  for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
    for (int iy = data.Lby; iy <= data.Uby; iy++) {
      for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
        for (int c = 0; c < ncomp; c++) {
          array(iz, iy, ix, c) = expected.direct(iz, iy, ix, c, ctx.offset);
        }
      }
    }
  }
}

template <typename Array>
void fill_all(Array& array, const ExchangeContext& ctx, const PicChunk::DataContainer& data,
              float64 base, int ncomp)
{
  ExpectedValues expected{ctx, data, base};
  auto           shape = array.shape();
  const int      nz    = static_cast<int>(shape[0]);
  const int      ny    = static_cast<int>(shape[1]);
  const int      nx    = static_cast<int>(shape[2]);

  int z_begin = ctx.has_dim[0] ? 0 : data.Lbz;
  int z_end   = ctx.has_dim[0] ? nz - 1 : data.Ubz;
  int y_begin = ctx.has_dim[1] ? 0 : data.Lby;
  int y_end   = ctx.has_dim[1] ? ny - 1 : data.Uby;
  int x_begin = ctx.has_dim[2] ? 0 : data.Lbx;
  int x_end   = ctx.has_dim[2] ? nx - 1 : data.Ubx;

  for (int iz = z_begin; iz <= z_end; iz++) {
    for (int iy = y_begin; iy <= y_end; iy++) {
      for (int ix = x_begin; ix <= x_end; ix++) {
        for (int c = 0; c < ncomp; c++) {
          array(iz, iy, ix, c) = expected.direct(iz, iy, ix, c, ctx.offset);
        }
      }
    }
  }
}

template <typename Array>
void verify_exchange(const Array& array, const ExchangeContext& ctx,
                     const PicChunk::DataContainer& data, float64 base, int ncomp)
{
  ExpectedValues expected{ctx, data, base};
  auto approx = [](float64 value) { return Catch::Approx(value).epsilon(1.0e-12).margin(1.0e-14); };

  auto      shape = array.shape();
  const int nz    = static_cast<int>(shape[0]);
  const int ny    = static_cast<int>(shape[1]);
  const int nx    = static_cast<int>(shape[2]);

  int z_begin = ctx.has_dim[0] ? 0 : data.Lbz;
  int z_end   = ctx.has_dim[0] ? nz - 1 : data.Ubz;
  int y_begin = ctx.has_dim[1] ? 0 : data.Lby;
  int y_end   = ctx.has_dim[1] ? ny - 1 : data.Uby;
  int x_begin = ctx.has_dim[2] ? 0 : data.Lbx;
  int x_end   = ctx.has_dim[2] ? nx - 1 : data.Ubx;

  for (int iz = z_begin; iz <= z_end; iz++) {
    for (int iy = y_begin; iy <= y_end; iy++) {
      for (int ix = x_begin; ix <= x_end; ix++) {
        for (int c = 0; c < ncomp; c++) {
          const float64 expected_value = expected.field(iz, iy, ix, c);
          INFO("index=" << iz << "," << iy << "," << ix << " comp=" << c);
          REQUIRE(array(iz, iy, ix, c) == approx(expected_value));
        }
      }
    }
  }
}

template <typename Array>
void verify_current_exchange(const Array& array, const ExchangeContext& ctx,
                             const PicChunk::DataContainer& data, float64 base, int ncomp)
{
  ExpectedValues expected{ctx, data, base};
  auto approx = [](float64 value) { return Catch::Approx(value).epsilon(1.0e-12).margin(1.0e-14); };

  auto      shape = array.shape();
  const int nz    = static_cast<int>(shape[0]);
  const int ny    = static_cast<int>(shape[1]);
  const int nx    = static_cast<int>(shape[2]);

  int z_begin = ctx.has_dim[0] ? 0 : data.Lbz;
  int z_end   = ctx.has_dim[0] ? nz - 1 : data.Ubz;
  int y_begin = ctx.has_dim[1] ? 0 : data.Lby;
  int y_end   = ctx.has_dim[1] ? ny - 1 : data.Uby;
  int x_begin = ctx.has_dim[2] ? 0 : data.Lbx;
  int x_end   = ctx.has_dim[2] ? nx - 1 : data.Ubx;

  for (int iz = z_begin; iz <= z_end; iz++) {
    for (int iy = y_begin; iy <= y_end; iy++) {
      for (int ix = x_begin; ix <= x_end; ix++) {
        for (int c = 0; c < ncomp; c++) {
          const float64 expected_value = expected.current(iz, iy, ix, c);
          INFO("index=" << iz << "," << iy << "," << ix << " comp=" << c);
          REQUIRE(array(iz, iy, ix, c) == approx(expected_value));
        }
      }
    }
  }
}

void run_boundary_exchange_test(const std::array<int, 3>& cdims, const nix::Dims3D& gdims,
                                const nix::Bool3D& has_dim, int order)
{
  auto ctx  = build_exchange_context(cdims, gdims, has_dim, order);
  auto data = ctx.chunk->get_internal_data();

  fill_interior(data.uf, ctx, data, 1000.0, 6);
  fill_all(data.uj, ctx, data, 2000.0, 4);

  exchange_boundary(*ctx.chunk, BoundaryEmf);
  exchange_boundary(*ctx.chunk, BoundaryCur);

  verify_exchange(data.uf, ctx, data, 1000.0, 6);
  verify_current_exchange(data.uj, ctx, data, 2000.0, 4);
}

void run_particle_exchange_test(const std::array<int, 3>& cdims, const nix::Dims3D& gdims,
                                const nix::Bool3D& has_dim, int order,
                                const std::array<int, 3>& dir)
{
  auto ctx  = build_exchange_context(cdims, gdims, has_dim, order);
  auto data = ctx.chunk->get_internal_data();

  data.up.resize(1);
  data.up[0]     = std::make_shared<ParticleType>(4, *ctx.chunk);
  data.up[0]->q  = 1.0;
  data.up[0]->m  = 1.0;
  data.up[0]->Np = 2;
  data.up[0]->xu.fill(0.0);
  data.up[0]->xv.fill(0.0);

  auto [xmin, xmax] = ctx.chunk->get_xrange();
  auto [ymin, ymax] = ctx.chunk->get_yrange();
  auto [zmin, zmax] = ctx.chunk->get_zrange();
  const int rank    = get_mpi_rank();

  //
  // Test Scenario
  // Keep one particle in-bounds and send one slightly out-of-bounds.
  // Direction is encoded by dir (z,y,x): each component is -1, 0, or +1.
  // Midpoints avoid accidental edge/corner sends when a dimension is inactive.
  // This exercises face and limited edge/corner cases without exploding combinations.
  // Particle IDs live in xu(:,6) and are used only for identity checks after sorting.
  //

  const float64 ymid = ctx.has_dim[1] ? 0.5 * (ymin + ymax) : 0.0;
  const float64 zmid = ctx.has_dim[0] ? 0.5 * (zmin + zmax) : 0.0;

  data.up[0]->xu(0, 0) = xmin + 0.25 * (xmax - xmin);
  data.up[0]->xu(0, 1) = ymid;
  data.up[0]->xu(0, 2) = zmid;
  data.up[0]->xu(0, 6) = static_cast<float64>(rank * 10 + 1);

  data.up[0]->xu(1, 0) = (dir[2] > 0)   ? xmax + 0.1
                         : (dir[2] < 0) ? xmin - 0.1
                                        : xmin + 0.75 * (xmax - xmin);
  data.up[0]->xu(1, 1) = (dir[1] > 0) ? ymax + 0.1 : (dir[1] < 0) ? ymin - 0.1 : ymid;
  data.up[0]->xu(1, 2) = (dir[0] > 0) ? zmax + 0.1 : (dir[0] < 0) ? zmin - 0.1 : zmid;
  data.up[0]->xu(1, 6) = static_cast<float64>(rank * 10 + 2);

  data.up[0]->count(0, data.up[0]->Np - 1, true, order);

  exchange_boundary(*ctx.chunk, BoundaryParticle);

  REQUIRE(data.up[0]->Np == 2);
  std::vector<int64_t> ids;
  ids.reserve(static_cast<size_t>(data.up[0]->Np));
  for (int ip = 0; ip < data.up[0]->Np; ip++) {
    ids.push_back(static_cast<int64_t>(data.up[0]->xu(ip, 6)));
  }
  std::sort(ids.begin(), ids.end());
  nix::ChunkMap    chunkmap(cdims[0], cdims[1], cdims[2]);
  std::vector<int> boundary(get_mpi_size() + 1);
  for (int i = 0; i <= get_mpi_size(); i++) {
    boundary[i] = i;
  }
  chunkmap.set_rank_boundary(boundary);
  const int            nb_cz    = chunkmap.get_neighbor_coord(ctx.coord[0], -dir[0], 0);
  const int            nb_cy    = chunkmap.get_neighbor_coord(ctx.coord[1], -dir[1], 1);
  const int            nb_cx    = chunkmap.get_neighbor_coord(ctx.coord[2], -dir[2], 2);
  const int            nb_id    = chunkmap.get_chunkid(nb_cz, nb_cy, nb_cx);
  const int            neighbor = chunkmap.get_rank(nb_id);
  std::vector<int64_t> expected = {rank * 10 + 1, neighbor * 10 + 2};
  std::sort(expected.begin(), expected.end());
  REQUIRE(ids == expected);
}

} // namespace

TEST_CASE("PicChunk boundary exchange 1D")
{
  const TestGrid grid = {{1, 1, 16}, {false, false, true}, {1, 1, 2}, 2};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int order = GENERATE(1, 2, 3, 4);
  run_boundary_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order);
}

TEST_CASE("PicChunk boundary exchange 2D")
{
  const TestGrid grid = {{1, 8, 8}, {false, true, true}, {1, 2, 2}, 4};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int order = GENERATE(1, 2, 3, 4);
  run_boundary_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order);
}

TEST_CASE("PicChunk boundary exchange 3D")
{
  const TestGrid grid = {{8, 8, 8}, {true, true, true}, {2, 2, 2}, 8};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int order = GENERATE(1, 2, 3, 4);
  run_boundary_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order);
}

TEST_CASE("PicChunk particle boundary exchange 1D")
{
  const TestGrid grid = {{1, 1, 16}, {false, false, true}, {1, 1, 2}, 2};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int                order = GENERATE(1, 2, 3, 4);
  const std::array<int, 3> dir =
      GENERATE(std::array<int, 3>{0, 0, 1}, std::array<int, 3>{0, 0, -1});
  run_particle_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order, dir);
}

TEST_CASE("PicChunk particle boundary exchange 2D")
{
  const TestGrid grid = {{1, 8, 8}, {false, true, true}, {1, 2, 2}, 4};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int                order = GENERATE(1, 2, 3, 4);
  const std::array<int, 3> dir = GENERATE(std::array<int, 3>{0, 0, 1}, std::array<int, 3>{0, 1, 0},
                                          std::array<int, 3>{0, 1, 1});
  run_particle_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order, dir);
}

TEST_CASE("PicChunk particle boundary exchange 3D")
{
  const TestGrid grid = {{8, 8, 8}, {true, true, true}, {2, 2, 2}, 8};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  const int                order = GENERATE(1, 2, 3, 4);
  const std::array<int, 3> dir = GENERATE(std::array<int, 3>{0, 0, 1}, std::array<int, 3>{0, 1, 0},
                                          std::array<int, 3>{1, 0, 0}, std::array<int, 3>{1, 1, 1});
  run_particle_exchange_test(grid.cdims, grid.gdims, grid.has_dim, order, dir);
}
