// -*- C++ -*-

#include "chunk.hpp"
#include "xtensor_halo3d.hpp"

#include "catch.hpp"

using namespace nix;

class TestChunk : public Chunk
{
public:
  static const bool* defaultHasDim()
  {
    static const bool has_dim[3] = {true, true, true};
    return has_dim;
  }

  TestChunk(const std::array<int, 3>& dims, int boundary_margin)
      : Chunk(dims.data(), defaultHasDim(), 0)
  {
    set_boundary_margin(boundary_margin);
    const int offset[3] = {0, 0, 0};
    const int gdims[3]  = {dims[0], dims[1], dims[2]};
    set_global_context(offset, gdims);
    set_coordinate(1.0, 1.0, 1.0);
  }

  int get_order() const
  {
    return 1;
  }

  void setup(json& config) override
  {
  }
};

void set_neighbor_ranks(TestChunk& chunk, int dirz, int diry, int dirx)
{
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        chunk.set_nb_rank(dz, dy, dx, MPI_PROC_NULL);
      }
    }
  }
  chunk.set_nb_rank(dirz, diry, dirx, 0);
}

TEST_CASE("XtensorHaloField3D pack/unpack +x")
{
  const std::array<int, 3> dims{4, 4, 4};
  const int                nb      = GENERATE(1, 2, 3);
  const int                ncomp   = 2;
  const float64            invalid = -1.0;

  TestChunk chunk(dims, nb);
  // Only +x neighbor is active for this test.
  set_neighbor_ranks(chunk, 0, 0, 1);

  const auto [Lbz, Ubz] = chunk.get_zbound();
  const auto [Lby, Uby] = chunk.get_ybound();
  const auto [Lbx, Ubx] = chunk.get_xbound();

  xt::xtensor<float64, 4> field(
      {static_cast<size_t>(dims[0] + 2 * nb), static_cast<size_t>(dims[1] + 2 * nb),
       static_cast<size_t>(dims[2] + 2 * nb), static_cast<size_t>(ncomp)});
  field.fill(invalid);

  for (int iz = Lbz; iz <= Ubz; ++iz) {
    for (int iy = Lby; iy <= Uby; ++iy) {
      for (int ix = Lbx; ix <= Ubx; ++ix) {
        for (int c = 0; c < ncomp; ++c) {
          field(iz, iy, ix, c) = static_cast<float64>(iz * 100000 + iy * 1000 + ix * 10 + c);
        }
      }
    }
  }

  auto mpibuf = std::make_shared<Chunk::MpiBuffer>();
  chunk.set_mpi_buffer(mpibuf, 0, 0, static_cast<int>(sizeof(float64) * ncomp));

  XtensorHaloField3D<TestChunk> halo(field, chunk);
  chunk.pack_bc_exchange(mpibuf, halo);

  const int iz      = 1;
  const int iy      = 1;
  const int ix      = 2;
  const int bufsize = mpibuf->bufsize(iz, iy, ix);
  std::memcpy(mpibuf->get_recv_buffer(iz, iy, ix), mpibuf->get_send_buffer(iz, iy, ix), bufsize);

  chunk.unpack_bc_exchange(mpibuf, halo);

  for (int iz = Lbz; iz <= Ubz; ++iz) {
    for (int iy = Lby; iy <= Uby; ++iy) {
      for (int layer = 0; layer < nb; ++layer) {
        const int src_ix = Ubx - nb + 1 + layer;
        const int dst_ix = Ubx + 1 + layer;
        for (int c = 0; c < ncomp; ++c) {
          const float64 expected = static_cast<float64>(iz * 100000 + iy * 1000 + src_ix * 10 + c);
          REQUIRE(field(iz, iy, dst_ix, c) == expected);
        }
      }
    }
  }
}

TEST_CASE("XtensorHaloCurrent3D pack/unpack +x adds to boundary")
{
  const std::array<int, 3> dims{4, 4, 4};
  const int                nb       = GENERATE(1, 2, 3);
  const int                ncomp    = 3;
  const float64            halo_val = 2.5;

  TestChunk chunk(dims, nb);
  // Only +x neighbor is active for this test.
  set_neighbor_ranks(chunk, 0, 0, 1);

  const auto [Lbz, Ubz] = chunk.get_zbound();
  const auto [Lby, Uby] = chunk.get_ybound();
  const auto [Lbx, Ubx] = chunk.get_xbound();

  xt::xtensor<float64, 4> current(
      {static_cast<size_t>(dims[0] + 2 * nb), static_cast<size_t>(dims[1] + 2 * nb),
       static_cast<size_t>(dims[2] + 2 * nb), static_cast<size_t>(ncomp)});
  current.fill(0.0);

  for (int iz = Lbz; iz <= Ubz; ++iz) {
    for (int iy = Lby; iy <= Uby; ++iy) {
      for (int ix = Ubx + 1; ix <= Ubx + nb; ++ix) {
        for (int c = 0; c < ncomp; ++c) {
          current(iz, iy, ix, c) = halo_val;
        }
      }
    }
  }

  auto mpibuf = std::make_shared<Chunk::MpiBuffer>();
  chunk.set_mpi_buffer(mpibuf, 0, 0, static_cast<int>(sizeof(float64) * ncomp));

  XtensorHaloCurrent3D<TestChunk> halo(current, chunk);
  chunk.pack_bc_exchange(mpibuf, halo);

  const int iz      = 1;
  const int iy      = 1;
  const int ix      = 2;
  const int bufsize = mpibuf->bufsize(iz, iy, ix);
  std::memcpy(mpibuf->get_recv_buffer(iz, iy, ix), mpibuf->get_send_buffer(iz, iy, ix), bufsize);

  chunk.unpack_bc_exchange(mpibuf, halo);

  for (int iz = Lbz; iz <= Ubz; ++iz) {
    for (int iy = Lby; iy <= Uby; ++iy) {
      for (int ix = Ubx - nb + 1; ix <= Ubx; ++ix) {
        for (int c = 0; c < ncomp; ++c) {
          REQUIRE(current(iz, iy, ix, c) == Approx(halo_val));
        }
      }
    }
  }
}

TEST_CASE("XtensorHaloParticle3D pack/unpack +x wraps particles")
{
  const std::array<int, 3> dims{4, 4, 4};
  const int                nb = GENERATE(1, 2, 3);

  TestChunk chunk(dims, nb);
  // Only +x neighbor is active for this test.
  set_neighbor_ranks(chunk, 0, 0, 1);

  ParticleVec particles;
  auto        species = std::make_shared<XtensorParticle>(4, chunk);
  species->Np         = 2;

  species->xu.fill(0.0);
  species->xu(0, 0)       = 0.5;
  species->xu(0, 1)       = 0.5;
  species->xu(0, 2)       = 0.5;
  const auto [xmin, xmax] = chunk.get_xrange();
  species->xu(1, 0)       = xmax + 0.1;
  species->xu(1, 1)       = 0.5;
  species->xu(1, 2)       = 0.5;

  species->count(0, species->Np - 1, true, chunk.get_order());
  particles.push_back(species);

  auto mpibuf = std::make_shared<Chunk::MpiBuffer>();
  chunk.set_mpi_buffer(mpibuf, 0, XtensorHaloParticle3D<TestChunk>::head_byte,
                       XtensorHaloParticle3D<TestChunk>::elem_byte);

  XtensorHaloParticle3D<TestChunk> halo(particles, chunk);
  chunk.pack_bc_exchange(mpibuf, halo);

  const int iz      = 1;
  const int iy      = 1;
  const int ix      = 2;
  const int bufsize = mpibuf->bufsize(iz, iy, ix);
  mpibuf->recvbuf.resize(mpibuf->sendbuf.size);
  std::memcpy(mpibuf->get_recv_buffer(iz, iy, ix), mpibuf->get_send_buffer(iz, iy, ix), bufsize);

  chunk.unpack_bc_exchange(mpibuf, halo);

  // post_unpack wraps and sorts particles, keeping the count unchanged.
  REQUIRE(species->Np == 2);

  int wrapped = 0;
  for (int ip = 0; ip < species->Np; ++ip) {
    if (species->xu(ip, 0) >= 0.0 && species->xu(ip, 0) < 4.0) {
      wrapped++;
    }
  }
  REQUIRE(wrapped == 2);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
