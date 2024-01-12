// -*- C++ -*-
#include "exchunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name

DEFINE_MEMBER(, ExChunk3D)
(const int dims[3], int id) : Chunk(dims, id), Ns(1), field_load(1.0)
{
  // check the minimum number of grids
  {
    bool is_valid = true;

    for (int dir = 0; dir < 3; dir++) {
      is_valid &= (dims[dir] >= Nb);
    }

    if (is_valid == false) {
      ERROR << tfm::format("Specified chunk dimensions are invalid.");
      ERROR << tfm::format("* Number of grid in x direction : %4d", dims[2]);
      ERROR << tfm::format("* Number of grid in y direction : %4d", dims[1]);
      ERROR << tfm::format("* Number of grid in z direction : %4d", dims[0]);
      ERROR << tfm::format("* Minimum number of grids       : %4d", Nb);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // initialize MPI buffer
  mpibufvec.resize(NumBoundaryMode);
  for (int i = 0; i < NumBoundaryMode; i++) {
    mpibufvec[i] = std::make_shared<MpiBuffer>();
  }

  // reset load
  this->load.resize(NumLoadMode);
  this->reset_load();
}

DEFINE_MEMBER(int64_t, get_size_byte)()
{
  int64_t size = 0;
  size += uf.size() * sizeof(float64);
  size += uj.size() * sizeof(float64);
  size += um.size() * sizeof(float64);
  // particle
  for (int is = 0; is < Ns; is++) {
    size += up[is]->get_size_byte();
  }
  // MPI Buffer
  for (int i = 0; i < NumBoundaryMode; i++) {
    size += mpibufvec[i]->get_size_byte();
  }
  return size;
}

DEFINE_MEMBER(void, setup_particle_mpi_buffer)(float64 fraction)
{
  int sizebyte[3][3][3];
  int zlen[3] = {1, dims[0], 1};
  int ylen[3] = {1, dims[1], 1};
  int xlen[3] = {1, dims[2], 1};

  int nppc = 0;
  for (int is = 0; is < Ns; is++) {
    nppc += up[is]->get_Np_active();
  }
  nppc /= (dims[0] * dims[1] * dims[2]);

  int byte_per_cell = static_cast<int>(nppc * fraction) * ParticleType::get_particle_size();

  for (int iz = 0; iz < 3; iz++) {
    for (int iy = 0; iy < 3; iy++) {
      for (int ix = 0; ix < 3; ix++) {
        sizebyte[iz][iy][ix] = zlen[iz] * ylen[iy] * xlen[ix] * byte_per_cell;
      }
    }
  }
  sizebyte[1][1][1] = 0;

  this->set_mpi_buffer(mpibufvec[BoundaryParticle], 0, sizeof(int) * Ns, sizebyte);
}

DEFINE_MEMBER(int, pack)(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count += Chunk::pack(buffer, count);
  count += memcpy_count(buffer, &field_load, sizeof(float64), count, 0);
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

DEFINE_MEMBER(int, unpack)(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count += Chunk::unpack(buffer, count);
  count += memcpy_count(&field_load, buffer, sizeof(float64), 0, count);
  count += memcpy_count(&Ns, buffer, sizeof(int), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);
  allocate(); // allocate memory for unpacking
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(uj.data(), buffer, uj.size() * sizeof(float64), 0, count);
  // particle (automatically allocate memory)
  up.resize(Ns);
  for (int is = 0; is < Ns; is++) {
    up[is] = std::make_shared<ParticleType>();
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

DEFINE_MEMBER(void, reset_load)()
{
  const int Ng = dims[0] * dims[1] * dims[2];

  load[LoadField]    = field_load;
  load[LoadParticle] = 0;
  for (int is = 0; is < up.size(); is++) {
    load[LoadParticle] += up[is]->Np / Ng;
  }
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

DEFINE_MEMBER(void, push_efd)(float64 delt)
{
  const float64 cflx = cc * delt / delx;
  const float64 cfly = cc * delt / dely;
  const float64 cflz = cc * delt / delz;

  // Ex
  for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb - 1; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 0) += (+cfly) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                             (-cflz) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4)) -
                             delt * uj(iz, iy, ix, 1);
      }
    }
  }

  // Ey
  for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb - 1; ix++) {
        uf(iz, iy, ix, 1) += (+cflz) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                             (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) -
                             delt * uj(iz, iy, ix, 2);
      }
    }
  }

  // Ez
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb - 1; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb - 1; ix++) {
        uf(iz, iy, ix, 2) += (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                             (-cfly) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3)) -
                             delt * uj(iz, iy, ix, 3);
      }
    }
  }
}

DEFINE_MEMBER(void, push_bfd)(float64 delt)
{
  const float64 cflx = cc * delt / delx;
  const float64 cfly = cc * delt / dely;
  const float64 cflz = cc * delt / delz;

  // Bx
  for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 3) += (-cfly) * (uf(iz, iy, ix, 2) - uf(iz, iy - 1, ix, 2)) +
                             (+cflz) * (uf(iz, iy, ix, 1) - uf(iz - 1, iy, ix, 1));
      }
    }
  }

  // By
  for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 4) += (-cflz) * (uf(iz, iy, ix, 0) - uf(iz - 1, iy, ix, 0)) +
                             (+cflx) * (uf(iz, iy, ix, 2) - uf(iz, iy, ix - 1, 2));
      }
    }
  }

  // Bz
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 5) += (-cflx) * (uf(iz, iy, ix, 1) - uf(iz, iy, ix - 1, 1)) +
                             (+cfly) * (uf(iz, iy, ix, 0) - uf(iz, iy - 1, ix, 0));
      }
    }
  }
}

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  const float64 rc = 1 / cc;

  for (int is = 0; is < Ns; is++) {
    auto ps = up[is];

    // loop over particle
    auto& xu = ps->xu;
    auto& xv = ps->xv;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 gam = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);
      float64 dt  = delt / gam;

      // substitute to temporary
      std::memcpy(&xv(ip, 0), &xu(ip, 0), ParticleType::get_particle_size());

      // update position
      xu(ip, 0) += xu(ip, 3) * dt;
      xu(ip, 1) += xu(ip, 4) * dt;
      xu(ip, 2) += xu(ip, 5) * dt;
    }

    // count
    this->count_particle(ps, 0, ps->Np - 1, true);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  constexpr int size   = Order + 1;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rc     = 1 / cc;
  const float64 rdx    = 1 / delx;
  const float64 rdy    = 1 / dely;
  const float64 rdz    = 1 / delz;
  const float64 dx2    = 0.5 * delx;
  const float64 dy2    = 0.5 * dely;
  const float64 dz2    = 0.5 * delz;
  const float64 ximin  = xlim[0] + dx2 * is_odd;
  const float64 xhmin  = xlim[0] + dx2 * is_odd - dx2;
  const float64 yimin  = ylim[0] + dy2 * is_odd;
  const float64 yhmin  = ylim[0] + dy2 * is_odd - dy2;
  const float64 zimin  = zlim[0] + dz2 * is_odd;
  const float64 zhmin  = zlim[0] + dz2 * is_odd - dz2;
  const float64 xigrid = ximin - dx2 * is_odd + dx2;
  const float64 xhgrid = xhmin - dx2 * is_odd + dx2;
  const float64 yigrid = yimin - dy2 * is_odd + dy2;
  const float64 yhgrid = yhmin - dy2 * is_odd + dy2;
  const float64 zigrid = zimin - dz2 * is_odd + dz2;
  const float64 zhgrid = zhmin - dz2 * is_odd + dz2;

  for (int is = 0; is < Ns; is++) {
    auto    ps  = up[is];
    float64 dt1 = 0.5 * ps->q / ps->m * delt;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wix[size] = {0};
      float64 whx[size] = {0};
      float64 wiy[size] = {0};
      float64 why[size] = {0};
      float64 wiz[size] = {0};
      float64 whz[size] = {0};

      float64 gam = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);
      float64 dt2 = dt1 * rc / gam;

      // grid indices
      int ix0 = digitize(xu(ip, 0), ximin, rdx);
      int hx0 = digitize(xu(ip, 0), xhmin, rdx);
      int iy0 = digitize(xu(ip, 1), yimin, rdy);
      int hy0 = digitize(xu(ip, 1), yhmin, rdy);
      int iz0 = digitize(xu(ip, 2), zimin, rdz);
      int hz0 = digitize(xu(ip, 2), zhmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xigrid + ix0 * delx, rdx, wix);
      shape<Order>(xu(ip, 0), xhgrid + hx0 * delx, rdx, whx);
      shape<Order>(xu(ip, 1), yigrid + iy0 * dely, rdy, wiy);
      shape<Order>(xu(ip, 1), yhgrid + hy0 * dely, rdy, why);
      shape<Order>(xu(ip, 2), zigrid + iz0 * delz, rdz, wiz);
      shape<Order>(xu(ip, 2), zhgrid + hz0 * delz, rdz, whz);

      //
      // calculate electromagnetic field at particle position
      //
      // * Ex => half-integer for x, full-integer for y, z
      // * Ey => half-integer for y, full-integer for z, x
      // * Ez => half-integer for z, full-integer for x, y
      // * Bx => half-integer for y, z, full-integer for x
      // * By => half-integer for z, x, full-integer for y
      // * Bz => half-integer for x, y, full-integer for z
      //
      ix0 += Lbx - (Order / 2);
      iy0 += Lby - (Order / 2);
      iz0 += Lbz - (Order / 2);
      hx0 += Lbx - (Order / 2);
      hy0 += Lby - (Order / 2);
      hz0 += Lbz - (Order / 2);

      float64 ex = interpolate3d<Order>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt1);
      float64 ey = interpolate3d<Order>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt1);
      float64 ez = interpolate3d<Order>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt1);
      float64 bx = interpolate3d<Order>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt2);
      float64 by = interpolate3d<Order>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt2);
      float64 bz = interpolate3d<Order>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt2);

      // push particle velocity
      push_boris(xu(ip, 3), xu(ip, 4), xu(ip, 5), ex, ey, ez, bx, by, bz);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  constexpr int size   = Order + 3;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dxdt  = delx / delt;
  const float64 dydt  = dely / delt;
  const float64 dzdt  = delz / delt;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 xmin  = xlim[0] + dx2 * is_odd;
  const float64 ymin  = ylim[0] + dy2 * is_odd;
  const float64 zmin  = zlim[0] + dz2 * is_odd;
  const float64 xgrid = xlim[0] + dx2;
  const float64 ygrid = ylim[0] + dy2;
  const float64 zgrid = zlim[0] + dz2;

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    auto    ps = up[is];
    float64 qs = ps->q;

    // loop over particle
    auto& xu = ps->xu;
    auto& xv = ps->xv;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 ss[2][3][size]           = {0};
      float64 cur[size][size][size][4] = {0};

      //
      // -*- weights before move -*-
      //
      // grid indices
      int ix0 = digitize(xv(ip, 0), xmin, rdx);
      int iy0 = digitize(xv(ip, 1), ymin, rdy);
      int iz0 = digitize(xv(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xv(ip, 0), xgrid + ix0 * delx, rdx, &ss[0][0][1]);
      shape<Order>(xv(ip, 1), ygrid + iy0 * dely, rdy, &ss[0][1][1]);
      shape<Order>(xv(ip, 2), zgrid + iz0 * delz, rdz, &ss[0][2][1]);

      //
      // -*- weights after move -*-
      //
      // grid indices
      int ix1 = digitize(xu(ip, 0), xmin, rdx);
      int iy1 = digitize(xu(ip, 1), ymin, rdy);
      int iz1 = digitize(xu(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xgrid + ix1 * delx, rdx, &ss[1][0][1 + ix1 - ix0]);
      shape<Order>(xu(ip, 1), ygrid + iy1 * dely, rdy, &ss[1][1][1 + iy1 - iy0]);
      shape<Order>(xu(ip, 2), zgrid + iz1 * delz, rdz, &ss[1][2][1 + iz1 - iz0]);

      //
      // -*- accumulate current via density decomposition -*-
      //
      esirkepov3d<Order>(dxdt, dydt, dzdt, ss, cur);

      ix0 += Lbx - (Order / 2) - 1;
      iy0 += Lby - (Order / 2) - 1;
      iz0 += Lbz - (Order / 2) - 1;
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            uj(iz, iy, ix, 0) += qs * cur[jz][jy][jx][0];
            uj(iz, iy, ix, 1) += qs * cur[jz][jy][jx][1];
            uj(iz, iy, ix, 2) += qs * cur[jz][jy][jx][2];
            uj(iz, iy, ix, 3) += qs * cur[jz][jy][jx][3];
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  constexpr int size   = Order + 1;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 xmin  = xlim[0] + dx2 * is_odd;
  const float64 ymin  = ylim[0] + dy2 * is_odd;
  const float64 zmin  = zlim[0] + dz2 * is_odd;
  const float64 xgrid = xlim[0] + dx2;
  const float64 ygrid = ylim[0] + dy2;
  const float64 zgrid = zlim[0] + dz2;

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    auto    ps = up[is];
    float64 ms = ps->m;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wx[size]                  = {0};
      float64 wy[size]                  = {0};
      float64 wz[size]                  = {0};
      float64 mom[size][size][size][11] = {0};

      // grid indices
      int ix0 = digitize(xu(ip, 0), xmin, rdx);
      int iy0 = digitize(xu(ip, 1), ymin, rdy);
      int iz0 = digitize(xu(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xgrid + ix0 * delx, rdx, wx);
      shape<Order>(xu(ip, 1), ygrid + iy0 * dely, rdy, wy);
      shape<Order>(xu(ip, 2), zgrid + iz0 * delz, rdz, wz);

      // deposit to local array (this step is not necessary for scalar version)
      for (int jz = 0; jz < size; jz++) {
        for (int jy = 0; jy < size; jy++) {
          for (int jx = 0; jx < size; jx++) {
            float64 ww = ms * wz[jz] * wy[jy] * wx[jx];
            float64 gm = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);

            //
            //  0: mass density
            //  1: tx-component T^{0,1} (x-component of momentum density)
            //  2: ty-component T^{0,2} (y-component of momentum density)
            //  3: tz-component T^{0,3} (z-component of momentum density)
            //  4: tt-component T^{0,0} (energy density / c)
            //  5: xx-component T^{1,1}
            //  6: yy-component T^{2,2}
            //  7: zz-component T^{3,3}
            //  8: xy-component T^{1,2}
            //  9: xz-component T^{1,3}
            // 10: yz-component T^{2,3}
            //
            mom[jz][jy][jx][0]  = ww;
            mom[jz][jy][jx][1]  = ww * xu(ip, 3);
            mom[jz][jy][jx][2]  = ww * xu(ip, 4);
            mom[jz][jy][jx][3]  = ww * xu(ip, 5);
            mom[jz][jy][jx][4]  = ww * gm * cc;
            mom[jz][jy][jx][5]  = ww * xu(ip, 3) * xu(ip, 3) / gm;
            mom[jz][jy][jx][6]  = ww * xu(ip, 4) * xu(ip, 4) / gm;
            mom[jz][jy][jx][7]  = ww * xu(ip, 5) * xu(ip, 5) / gm;
            mom[jz][jy][jx][8]  = ww * xu(ip, 3) * xu(ip, 4) / gm;
            mom[jz][jy][jx][9]  = ww * xu(ip, 3) * xu(ip, 5) / gm;
            mom[jz][jy][jx][10] = ww * xu(ip, 4) * xu(ip, 5) / gm;
          }
        }
      }

      // deposit to global array
      ix0 += Lbx - (Order / 2);
      iy0 += Lby - (Order / 2);
      iz0 += Lbz - (Order / 2);
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            for (int k = 0; k < 11; k++) {
              um(iz, iy, ix, is, k) += mom[jz][jy][jx][k];
            }
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, set_boundary_begin)(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<ThisType>(uf, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<ThisType>(uj, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<ThisType>(um, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<ThisType>(up, *this);
    this->inject_particle(up);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }
}

DEFINE_MEMBER(void, set_boundary_end)(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<ThisType>(uf, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<ThisType>(uj, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<ThisType>(um, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<ThisType>(up, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }

  // physical boundary for field
  this->set_boundary_field(mode);
}

template class ExChunk3D<1>;
template class ExChunk3D<2>;
template class ExChunk3D<3>;
template class ExChunk3D<4>;

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
