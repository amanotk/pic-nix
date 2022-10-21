// -*- C++ -*-
#include "exchunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name

DEFINE_MEMBER(, ExChunk3D)(const int dims[3], const int id) : Chunk(dims, id), Ns(1)
{
  // initialize MPI buffer
  mpibufvec.resize(NumBoundaryMode);
  for (int i = 0; i < NumBoundaryMode; i++) {
    mpibufvec[i] = std::make_shared<MpiBuffer>();
  }

  // reset load
  this->load.resize(NumLoadMode);
  this->reset_load();
}

DEFINE_MEMBER(int, pack)(void* buffer, const int address)
{
  using nix::memcpy_count;

  int count = address;

  count += Chunk::pack(buffer, count);
  count += memcpy_count(buffer, &Ns, sizeof(int), count, 0);
  count += memcpy_count(buffer, &cc, sizeof(float64), count, 0);
  count += memcpy_count(buffer, uf.data(), uf.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, uj.data(), uj.size() * sizeof(float64), count, 0);
  // particle
  for (int is = 0; is < Ns; is++) {
    count += up[is]->pack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void* buffer, const int address)
{
  using nix::memcpy_count;

  int count = address;

  count += Chunk::unpack(buffer, count);
  count += memcpy_count(&Ns, buffer, sizeof(int), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);
  allocate(); // allocate memory for unpacking
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(uj.data(), buffer, uj.size() * sizeof(float64), 0, count);
  // particle (automatically allocate memory)
  up.resize(Ns);
  for (int is = 0; is < Ns; is++) {
    up[is] = std::make_shared<Particle>();
    count += up[is]->unpack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(void, allocate)()
{
  size_t nz = dims[0] + 2 * Nb;
  size_t ny = dims[1] + 2 * Nb;
  size_t nx = dims[2] + 2 * Nb;
  size_t ns = Ns;

  // memory allocation
  uf.resize({nz, ny, nx, 6});
  uj.resize({nz, ny, nx, 4});
  um.resize({nz, ny, nx, ns, 11});
  uf.fill(0);
  uj.fill(0);
  um.fill(0);
}

DEFINE_MEMBER(int, pack_diagnostic)(const int mode, void* buffer, const int address)
{
  switch (mode) {
  case DiagnosticLoad:
    return this->pack_diagnostic_load(buffer, address);
    break;
  case DiagnosticX:
    return this->pack_diagnostic_coord(buffer, address, 2);
    break;
  case DiagnosticY:
    return this->pack_diagnostic_coord(buffer, address, 1);
    break;
  case DiagnosticZ:
    return this->pack_diagnostic_coord(buffer, address, 0);
    break;
  case DiagnosticEmf:
    return this->pack_diagnostic_field(buffer, address, uf);
    break;
  case DiagnosticCur:
    return this->pack_diagnostic_field(buffer, address, uj);
    break;
  case DiagnosticMom: {
    return this->pack_diagnostic_field(buffer, address, um);
  } break;
  default:
    break;
  }

  // pack a specific particle species
  if (mode >= DiagnosticParticle && mode < DiagnosticCustom) {
    int is = mode - DiagnosticParticle;
    return this->pack_diagnostic_particle(buffer, address, up[is]);
  }

  return 0;
}

DEFINE_MEMBER(void, setup)(json& config)
{
  // a derived class should implement it
}

DEFINE_MEMBER(void, get_energy)(float64& efd, float64& bfd, float64 particle[])
{
  // clear
  efd = 0.0;
  bfd = 0.0;
  std::fill(particle, particle + Ns, 0.0);

  // electromagnetic energy
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        efd +=
            0.5 * (uf(iz, iy, ix, 0) * uf(iz, iy, ix, 0) + uf(iz, iy, ix, 1) * uf(iz, iy, ix, 1) +
                   uf(iz, iy, ix, 2) * uf(iz, iy, ix, 2));
        bfd +=
            0.5 * (uf(iz, iy, ix, 3) * uf(iz, iy, ix, 3) + uf(iz, iy, ix, 4) * uf(iz, iy, ix, 4) +
                   uf(iz, iy, ix, 5) * uf(iz, iy, ix, 5));
      }
    }
  }
  efd *= delx * dely * delz;
  bfd *= delx * dely * delz;

  // particle energy for each species
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        for (int is = 0; is < Ns; is++) {
          // rest mass energy is subtracted
          particle[is] += um(iz, iy, ix, is, 4) * cc - um(iz, iy, ix, is, 0) * cc * cc;
        }
      }
    }
  }
}

DEFINE_MEMBER(void, get_diverror)(float64& efd, float64& bfd)
{
  const float64 rdx = 1 / delx;
  const float64 rdy = 1 / dely;
  const float64 rdz = 1 / delz;

  efd = 0;
  bfd = 0;

  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        // div(E) - rho
        efd += (uf(iz, iy, ix + 1, 0) - uf(iz, iy, ix, 0)) * rdx +
               (uf(iz, iy + 1, ix, 1) - uf(iz, iy, ix, 1)) * rdy +
               (uf(iz + 1, iy, ix, 2) - uf(iz, iy, ix, 2)) * rdz - uj(iz, iy, ix, 0);
        // div(B)
        bfd += (uf(iz, iy, ix, 3) - uf(iz, iy, ix - 1, 3)) * rdx +
               (uf(iz, iy, ix, 4) - uf(iz, iy - 1, ix, 4)) * rdy +
               (uf(iz, iy, ix, 5) - uf(iz - 1, iy, ix, 5)) * rdz;
      }
    }
  }
}

DEFINE_MEMBER(void, push_efd)(const float64 delt)
{
  const float64 cflx = cc * delt / delx;
  const float64 cfly = cc * delt / dely;
  const float64 cflz = cc * delt / delz;

  float64 etime = nix::wall_clock();

  for (int iz = Lbz - 1; iz <= Ubz; iz++) {
    for (int iy = Lby - 1; iy <= Uby; iy++) {
      for (int ix = Lbx - 1; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 0) += (+cfly) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                             (-cflz) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4)) -
                             delt * uj(iz, iy, ix, 1);
        uf(iz, iy, ix, 1) += (+cflz) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                             (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) -
                             delt * uj(iz, iy, ix, 2);
        uf(iz, iy, ix, 2) += (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                             (-cfly) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3)) -
                             delt * uj(iz, iy, ix, 3);
      }
    }
  }

  // store computation time
  this->load[LoadEmf] += nix::wall_clock() - etime;
}

DEFINE_MEMBER(void, push_bfd)(const float64 delt)
{
  const float64 cflx = cc * delt / delx;
  const float64 cfly = cc * delt / dely;
  const float64 cflz = cc * delt / delz;

  float64 etime = nix::wall_clock();

  for (int iz = Lbz; iz <= Ubz + 1; iz++) {
    for (int iy = Lby; iy <= Uby + 1; iy++) {
      for (int ix = Lbx; ix <= Ubx + 1; ix++) {
        uf(iz, iy, ix, 3) += (-cfly) * (uf(iz, iy, ix, 2) - uf(iz, iy - 1, ix, 2)) +
                             (+cflz) * (uf(iz, iy, ix, 1) - uf(iz - 1, iy, ix, 1));
        uf(iz, iy, ix, 4) += (-cflz) * (uf(iz, iy, ix, 0) - uf(iz - 1, iy, ix, 0)) +
                             (+cflx) * (uf(iz, iy, ix, 2) - uf(iz, iy, ix - 1, 2));
        uf(iz, iy, ix, 5) += (-cflx) * (uf(iz, iy, ix, 1) - uf(iz, iy, ix - 1, 1)) +
                             (+cfly) * (uf(iz, iy, ix, 0) - uf(iz, iy - 1, ix, 0));
      }
    }
  }

  // store computation time
  this->load[LoadEmf] += nix::wall_clock() - etime;
}

DEFINE_MEMBER(void, set_boundary_begin)(const int mode)
{
  // physical boundary
  this->set_boundary_physical(mode);

  switch (mode) {
  case BoundaryEmf:
    this->begin_bc_exchange(mpibufvec[mode], uf);
    break;
  case BoundaryCur:
    this->begin_bc_exchange(mpibufvec[mode], uj, true);
    break;
  case BoundaryMom:
    this->begin_bc_exchange(mpibufvec[mode], um, true);
    break;
  case BoundaryParticle:
    this->inject_particle(up);
    this->begin_bc_exchange(mpibufvec[mode], up);
    break;
  default:
    ERRORPRINT("No such boundary mode exists!\n");
    break;
  }
}

DEFINE_MEMBER(void, set_boundary_end)(const int mode)
{
  switch (mode) {
  case BoundaryEmf:
    this->end_bc_exchange(mpibufvec[mode], uf);
    break;
  case BoundaryCur:
    this->end_bc_exchange(mpibufvec[mode], uj, true);
    break;
  case BoundaryMom:
    this->end_bc_exchange(mpibufvec[mode], um, true);
    break;
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
