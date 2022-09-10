// -*- C++ -*-
#include "exchunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type ExChunk3D<Nb>::name

DEFINE_MEMBER(, ExChunk3D)(const int dims[3], const int id) : Chunk(dims, id), Ns(1)
{
  // initialize MPI buffer
  mpibufvec.resize(NumBoundaryMode);
  for (int i = 0; i < NumBoundaryMode; i++) {
    mpibufvec[i] = std::make_shared<MpiBuffer>();
  }
}

DEFINE_MEMBER(, ~ExChunk3D)()
{
}

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk::pack(buffer, count);
  count += memcpy_count(buffer, uf.data(), uf.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, uj.data(), uj.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, &cc, sizeof(float64), count, 0);
  // particle
  for (int is = 0; is < Ns; is++) {
    count += up[is]->pack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk::unpack(buffer, count);
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(uj.data(), buffer, uj.size() * sizeof(float64), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);
  // particle
  for (int is = 0; is < Ns; is++) {
    count += up[is]->unpack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(void, allocate_memory)(const int Np, const int Ns)
{
  this->Ns = Ns;

  size_t nz = dims[0] + 2 * Nb;
  size_t ny = dims[1] + 2 * Nb;
  size_t nx = dims[2] + 2 * Nb;
  size_t ns = Ns;
  size_t np = Np / (nz * ny * nx); // average particle per cell

  // memory allocation
  uf.resize({nz, ny, nx, 6});
  uj.resize({nz, ny, nx, 4});
  um.resize({nz, ny, nx, ns, 10});
  uf.fill(0);
  uj.fill(0);
  um.fill(0);

  for (int is = 0; is < ns; is++) {
    up.push_back(std::make_shared<Particle>(Np, nz * ny * nx));
  }

  // allocate MPI buffer
  this->set_mpi_buffer(mpibufvec[BoundaryEmf], 0, sizeof(float64) * 6);
  this->set_mpi_buffer(mpibufvec[BoundaryCur], 0, sizeof(float64) * 4);
  this->set_mpi_buffer(mpibufvec[BoundaryMom], 0, sizeof(float64) * Ns * 10);
  this->set_mpi_buffer(mpibufvec[BoundaryParticle], sizeof(int), sizeof(float64) * Ns * np * 7);
}

DEFINE_MEMBER(int, pack_diagnostic)(const int mode, void *buffer, const int address)
{
  switch (mode) {
  case PackEmf:
    return pack_diagnostic_field(buffer, address, uf);
    break;
  case PackCur:
    return pack_diagnostic_field(buffer, address, uj);
    break;
  case PackMom: {
    // reshape into 4D array
    std::vector<size_t> shape(4);
    shape[0]   = um.shape(0);
    shape[1]   = um.shape(1);
    shape[2]   = um.shape(2);
    shape[3]   = um.shape(3) * um.shape(4);
    T_field vv = xt::adapt(um.data(), um.size(), xt::no_ownership(), shape);
    return pack_diagnostic_field(buffer, address, vv);
  } break;
  case PackParticle:
    ERRORPRINT("Not implemented yet\n");
    break;
  default:
    ERRORPRINT("No such pack mode exists\n");
    break;
  }
  return 0;
}

DEFINE_MEMBER(int, pack_diagnostic_field)(void *buffer, const int address, T_field &u)
{
  size_t size = dims[2] * dims[1] * dims[0] * u.shape(3);

  if (buffer == nullptr) {
    return sizeof(float64) * size;
  }

  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);
  auto vv = xt::view(u, Iz, Iy, Ix, xt::all());

  // packing
  char *ptr = &static_cast<char *>(buffer)[address];
  std::copy(vv.begin(), vv.end(), ptr);

  return sizeof(float64) * size;
}

DEFINE_MEMBER(void, setup)(const float64 cc, const float64 delh, const int offset[3])
{
  // this->set_coordinate(delh, offset);

  // speed of light
  this->cc = cc;
}

DEFINE_MEMBER(void, push_efd)(const float64 delt)
{
  const float64 delh = this->delh;
  const float64 cfl  = cc * delt / delh;

  float64 etime = common::etime();

  for (int iz = Lbz - 1; iz <= Ubz; iz++) {
    for (int iy = Lby - 1; iy <= Uby; iy++) {
      for (int ix = Lbx - 1; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 0) += (+cfl) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                             (-cfl) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4)) -
                             delt * uj(iz, iy, ix, 1);
        uf(iz, iy, ix, 1) += (+cfl) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                             (-cfl) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) -
                             delt * uj(iz, iy, ix, 2);
        uf(iz, iy, ix, 2) += (+cfl) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                             (-cfl) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3)) -
                             delt * uj(iz, iy, ix, 3);
      }
    }
  }

  // store computation time
  this->load += common::etime() - etime;
}

DEFINE_MEMBER(void, push_mfd)(const float64 delt)
{
  const float64 delh = this->delh;
  const float64 cfl  = cc * delt / delh;

  float64 etime = common::etime();

  for (int iz = Lbz; iz <= Ubz + 1; iz++) {
    for (int iy = Lby; iy <= Uby + 1; iy++) {
      for (int ix = Lbx; ix <= Ubx + 1; ix++) {
        uf(iz, iy, ix, 3) += (-cfl) * (uf(iz, iy, ix, 2) - uf(iz, iy - 1, ix, 2)) +
                             (+cfl) * (uf(iz, iy, ix, 1) - uf(iz - 1, iy, ix, 1));
        uf(iz, iy, ix, 4) += (-cfl) * (uf(iz, iy, ix, 0) - uf(iz - 1, iy, ix, 0)) +
                             (+cfl) * (uf(iz, iy, ix, 2) - uf(iz, iy, ix - 1, 2));
        uf(iz, iy, ix, 5) += (-cfl) * (uf(iz, iy, ix, 1) - uf(iz, iy, ix - 1, 1)) +
                             (+cfl) * (uf(iz, iy, ix, 0) - uf(iz, iy - 1, ix, 0));
      }
    }
  }

  // store computation time
  this->load += common::etime() - etime;
}

DEFINE_MEMBER(void, set_boundary_begin)(const int mode)
{
  auto Ia = xt::all();

  // physical boundary
  this->set_boundary_physical(mode);

  switch (mode) {
  case BoundaryEmf:
    this->begin_bc_exchange(mpibufvec[mode], uf);
    break;
  case BoundaryCur:
    this->begin_bc_exchange(mpibufvec[mode], uj);
    break;
  case BoundaryMom: {
    // reshape into 4D array
    std::vector<size_t> shape(4);
    shape[0]   = um.shape(0);
    shape[1]   = um.shape(1);
    shape[2]   = um.shape(2);
    shape[3]   = um.shape(3) * um.shape(4);
    T_field vv = xt::adapt(um.data(), um.size(), xt::no_ownership(), shape);
    this->begin_bc_exchange(mpibufvec[mode], vv);
  } break;
  case BoundaryParticle:
    // particle injection should be placed here
    this->begin_bc_exchange(mpibufvec[mode], up);
    break;
  default:
    ERRORPRINT("No such boundary mode exists!\n");
    break;
  }
}

DEFINE_MEMBER(void, set_boundary_end)(const int mode)
{
  auto Ia = xt::all();

  switch (mode) {
  case BoundaryEmf:
    this->end_bc_exchange(mpibufvec[mode], uf, false);
    break;
  case BoundaryCur:
    this->end_bc_exchange(mpibufvec[mode], uj, true); // append
    break;
  case BoundaryMom: {
    // reshape into 4D array
    std::vector<size_t> shape(4);
    shape[0]   = um.shape(0);
    shape[1]   = um.shape(1);
    shape[2]   = um.shape(2);
    shape[3]   = um.shape(3) * um.shape(4);
    T_field vv = xt::adapt(um.data(), um.size(), xt::no_ownership(), shape);
    this->end_bc_exchange(mpibufvec[mode], vv, true); // append
  } break;
  case BoundaryParticle:
    this->end_bc_exchange(mpibufvec[mode], up);
    break;
  default:
    ERRORPRINT("No such boundary mode exists!\n");
    break;
  }
}

// implementation for specific shape functions
#include "exchunk3d_1st.cpp"

template class ExChunk3D<1>;
// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
