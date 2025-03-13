// -*- C++ -*-
#include "exchunk3d.hpp"

#include "engine.hpp"

#define DEFINE_MEMBER(type, name) type ExChunk3D::name

DEFINE_MEMBER(, ExChunk3D)
(const int dims[3], int id) : Chunk(dims, id), Ns(1)
{
  // check dimension
  {
    bool is_3d = dims[0] >= 2 && dims[1] >= 2 && dims[2] >= 2;
    bool is_2d = dims[0] == 1 && dims[1] >= 2 && dims[2] >= 2;
    bool is_1d = dims[0] == 1 && dims[1] == 1 && dims[2] >= 2;

    if (is_1d != true && is_2d != true && is_3d != true) {
      ERROR << tfm::format("Invalid dimension: %d %d %d", dims[0], dims[1], dims[2]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    } else if (is_1d == true) {
      dimension = 1;
    } else if (is_2d == true) {
      dimension = 2;
    } else if (is_3d == true) {
      dimension = 3;
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

DEFINE_MEMBER(int, pack)(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count = Chunk::pack(buffer, count);

  count += memcpy_count(buffer, &order, sizeof(int), count, 0);
  count += memcpy_count(buffer, &Ns, sizeof(int), count, 0);
  count += memcpy_count(buffer, &cc, sizeof(float64), count, 0);
  count += memcpy_count(buffer, uf.data(), uf.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, uj.data(), uj.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, ff.data(), ff.size() * sizeof(float64), count, 0);
  // particle
  for (int is = 0; is < Ns; is++) {
    count = up[is]->pack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count = Chunk::unpack(buffer, count);

  count += memcpy_count(&order, buffer, sizeof(int), 0, count);
  count += memcpy_count(&Ns, buffer, sizeof(int), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);
  allocate(); // allocate memory for unpacking
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(uj.data(), buffer, uj.size() * sizeof(float64), 0, count);
  count += memcpy_count(ff.data(), buffer, ff.size() * sizeof(float64), 0, count);
  // particle (automatically allocate memory)
  up.resize(Ns);
  for (int is = 0; is < Ns; is++) {
    up[is] = std::make_shared<ParticleType>();
    count  = up[is]->unpack(buffer, count);
  }

  return count;
}

DEFINE_MEMBER(void, allocate)()
{
  size_t nz = dims[0] + 2 * boundary_margin;
  size_t ny = dims[1] + 2 * boundary_margin;
  size_t nx = dims[2] + 2 * boundary_margin;
  size_t ns = Ns;

  // memory allocation
  uf.resize({nz, ny, nx, 6});
  uj.resize({nz, ny, nx, 4});
  um.resize({nz, ny, nx, ns, 14});
  ff.resize({nz, ny, nx, 3, 6});
  uf.fill(0);
  uj.fill(0);
  um.fill(0);
  ff.fill(0);
}

DEFINE_MEMBER(void, reset_load)()
{
  const int Ng = dims[0] * dims[1] * dims[2];

  load[LoadField]    = option.value("cell_load", 1.0);
  load[LoadParticle] = 0;
  for (int is = 0; is < up.size(); is++) {
    load[LoadParticle] += up[is]->Np / Ng;
  }
}

DEFINE_MEMBER(void, setup)(json& config)
{
  auto opt = config["option"];

  // vectorization mode
  {
    std::vector<std::string> valid_mode = {"scalar", "vector"};

    auto vectorization = opt["vectorization"];
    bool is_success    = true;

    // default
    if (vectorization.is_null() == true) {
      vectorization = "scalar";
    }

    if (vectorization.is_string() == true) {
      // string type
      std::string vector_or_scalar = vectorization.get<std::string>();
      if (vector_or_scalar == "scalar") {
        option["vectorization"] = {{"position", "scalar"},
                                   {"velocity", "scalar"},
                                   {"current", "scalar"},
                                   {"moment", "scalar"}};
      } else if (vector_or_scalar == "vector") {
        option["vectorization"] = {{"position", "scalar"},
                                   {"velocity", "vector"},
                                   {"current", "vector"},
                                   {"moment", "vector"}};
      } else {
        is_success = false;
      }
    } else if (vectorization.is_object() == true) {
      // assume specific vectorization mode for each variable
      for (auto& key : {"position", "velocity", "current", "moment"}) {
        std::string mode = vectorization.value(key, "scalar");

        bool is_valid_mode =
            std::find(valid_mode.begin(), valid_mode.end(), mode) != valid_mode.end();
        if (is_valid_mode == true) {
          option["vectorization"][key] = mode;
        } else {
          is_success = false;
        }
      }
    } else {
      is_success = false;
    }

    if (is_success == false) {
      ERROR << tfm::format("Invalid vectorization mode (see below):");
      std::cerr << std::setw(2) << vectorization << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // order of shape function
  {
    order = opt.value("order", 2);

    int  nb    = (order + 3) / 2; // required boundary margin
    bool is_1d = dimension == 1 && dims[2] >= nb;
    bool is_2d = dimension == 2 && dims[1] >= nb && dims[2] >= nb;
    bool is_3d = dimension == 3 && dims[0] >= nb && dims[1] >= nb && dims[2] >= nb;

    if (is_1d == false && is_2d == false && is_3d == false) {
      ERROR << tfm::format("Number of grid points smaller than the minimum (%2d) "
                           "for the chosen shape order (%2d) of shape function ",
                           nb, order);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    option["order"] = order;
    this->set_boundary_margin(nb);
  }

  // interpolation
  {
    auto interpolation = opt.value("interpolation", "MC");

    if (interpolation == engine::InterpName[engine::InterpMC]) {
      option["interpolation"] = engine::InterpMC;
    } else if (interpolation == engine::InterpName[engine::InterpWT]) {
      option["interpolation"] = engine::InterpWT;
    } else {
      ERROR << tfm::format("Invalid interpolation: %s", interpolation);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // seed for random number generator
  {
    auto seed_type = opt.value("seed_type", "random");

    if (seed_type == "random") {
      option["random_seed"] = std::random_device()();
    } else if (seed_type == "fixed") {
      option["random_seed"] = this->myid; // chunk ID
    } else {
      ERROR << tfm::format("Invalid seed type: %s", seed_type);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // misc
  {
    option["friedman"]     = opt.value("friedman", 0.0);
    option["cell_load"]    = opt.value("cell_load", 1.0);
    option["buffer_ratio"] = opt.value("buffer_ratio", 0.2);
  }
}

DEFINE_MEMBER(void, setup_friedman_filter)()
{
  const int Nb = boundary_margin;

  // initialize electric field for Friedman filter
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        for (int dir = 0; dir < 3; dir++) {
          ff(iz, iy, ix, 0, dir) = uf(iz, iy, ix, dir);
          ff(iz, iy, ix, 1, dir) = uf(iz, iy, ix, dir);
          ff(iz, iy, ix, 2, dir) = uf(iz, iy, ix, dir);
        }
      }
    }
  }
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
  const int     Nb    = boundary_margin;
  const float64 theta = option["friedman"].get<float64>();
  const float64 cflx  = cc * delt / delx;
  const float64 cfly  = cc * delt / dely;
  const float64 cflz  = cc * delt / delz;

  // update for Friedman filter first (boundary condition has already been set)
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        // Ex
        ff(iz, iy, ix, 2, 0) = ff(iz, iy, ix, 1, 0) + theta * ff(iz, iy, ix, 2, 0);
        ff(iz, iy, ix, 1, 0) = uf(iz, iy, ix, 0);
        // Ey
        ff(iz, iy, ix, 2, 1) = ff(iz, iy, ix, 1, 1) + theta * ff(iz, iy, ix, 2, 1);
        ff(iz, iy, ix, 1, 1) = uf(iz, iy, ix, 1);
        // Ez
        ff(iz, iy, ix, 2, 2) = ff(iz, iy, ix, 1, 2) + theta * ff(iz, iy, ix, 2, 2);
        ff(iz, iy, ix, 1, 2) = uf(iz, iy, ix, 2);
      }
    }
  }

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
  const int     Nb    = boundary_margin;
  const float64 theta = option["friedman"].get<float64>();
  const float64 A     = 1 + 0.5 * theta;
  const float64 B     = -theta * (1 - 0.5 * theta);
  const float64 C     = 0.5 * theta * (1 - theta) * (1 - theta);
  const float64 cflx  = cc * delt / delx;
  const float64 cfly  = cc * delt / dely;
  const float64 cflz  = cc * delt / delz;

  // update for Friedman filter first (boundary condition has already been set)
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        ff(iz, iy, ix, 0, 0) =
            A * uf(iz, iy, ix, 0) + B * ff(iz, iy, ix, 1, 0) + C * ff(iz, iy, ix, 2, 0);
        ff(iz, iy, ix, 0, 1) =
            A * uf(iz, iy, ix, 1) + B * ff(iz, iy, ix, 1, 1) + C * ff(iz, iy, ix, 2, 1);
        ff(iz, iy, ix, 0, 2) =
            A * uf(iz, iy, ix, 2) + B * ff(iz, iy, ix, 1, 2) + C * ff(iz, iy, ix, 2, 2);
      }
    }
  }

  // Bx
  for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 3) += (-cfly) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy - 1, ix, 0, 2)) +
                             (+cflz) * (ff(iz, iy, ix, 0, 1) - ff(iz - 1, iy, ix, 0, 1));
      }
    }
  }

  // By
  for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 4) += (-cflz) * (ff(iz, iy, ix, 0, 0) - ff(iz - 1, iy, ix, 0, 0)) +
                             (+cflx) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy, ix - 1, 0, 2));
      }
    }
  }

  // Bz
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
      for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
        uf(iz, iy, ix, 5) += (-cflx) * (ff(iz, iy, ix, 0, 1) - ff(iz, iy, ix - 1, 0, 1)) +
                             (+cfly) * (ff(iz, iy, ix, 0, 0) - ff(iz, iy - 1, ix, 0, 0));
      }
    }
  }
}

DEFINE_MEMBER(bool, set_boundary_probe)(int mode, bool wait)
{
  if (mode == BoundaryParticle) {
    if (wait == true) {
      while (this->probe_bc_exchange(mpibufvec[mode]) == false) {
        // wait for completion
      }
      return true;
    } else {
      // immediately return status
      return this->probe_bc_exchange(mpibufvec[mode]);
    }
  }

  return true;
}

DEFINE_MEMBER(void, set_boundary_pack)(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<ThisType>(uf, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<ThisType>(uj, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<ThisType>(um, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<ThisType>(up, *this);
    this->inject_particle(up);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }
}

DEFINE_MEMBER(void, set_boundary_unpack)(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<ThisType>(uf, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<ThisType>(uj, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<ThisType>(um, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<ThisType>(up, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }

  // physical boundary for field
  this->set_boundary_field(mode);
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
}

DEFINE_MEMBER(void, count_particle)(ParticlePtr particle, int Lbp, int Ubp, bool reset)
{
  // notice the half-grid offset of cell boundaries for odd-order shape functions
  const int     is_odd        = (order % 2 == 1) ? 1 : 0;
  const int     out_of_bounds = particle->Ng;
  const float64 xmin          = xlim[0] - 0.5 * delx * is_odd;
  const float64 ymin          = ylim[0] - 0.5 * dely * is_odd;
  const float64 zmin          = zlim[0] - 0.5 * delz * is_odd;

  // parameters for sort by cell
  int     stride_x = 1;
  int     stride_y = stride_x * (Ubx - Lbx + 2);
  int     stride_z = stride_y * (Uby - Lby + 2);
  float64 rdx      = 1 / delx;
  float64 rdy      = 1 / dely;
  float64 rdz      = 1 / delz;

  // reset count
  if (reset) {
    particle->reset_count();
  }

  auto& xu = particle->xu;
  for (int ip = Lbp; ip <= Ubp; ip++) {
    int ix = digitize(xu(ip, 0), xmin, rdx);
    int iy = digitize(xu(ip, 1), ymin, rdy);
    int iz = digitize(xu(ip, 2), zmin, rdz);
    int ii = iz * stride_z + iy * stride_y + ix * stride_x;

    // take care out-of-bounds particles
    ii = (xu(ip, 0) < xlim[0] || xu(ip, 0) >= xlim[1]) ? out_of_bounds : ii;
    ii = (xu(ip, 1) < ylim[0] || xu(ip, 1) >= ylim[1]) ? out_of_bounds : ii;
    ii = (xu(ip, 2) < zlim[0] || xu(ip, 2) >= zlim[1]) ? out_of_bounds : ii;

    particle->increment(ip, ii);
  }
}

DEFINE_MEMBER(void, sort_particle)(ParticleVec& particle)
{
  for (int is = 0; is < particle.size(); is++) {
    count_particle(particle[is], 0, particle[is]->Np - 1, true);
    particle[is]->sort();
  }
}

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  auto      mode = option["vectorization"]["position"].get<std::string>();
  const int V    = "vector" == mode;

  engine::PositionEngine<InternalData> position;
  position(V, get_internal_data(), delt);

  // apply boundary condition and count particles
  for (int is = 0; is < Ns; is++) {
    set_boundary_particle(up[is], 0, up[is]->Np - 1, is);
    count_particle(up[is], 0, up[is]->Np - 1, true);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  auto      mode = option["vectorization"]["velocity"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;
  const int P    = engine::PusherBoris;
  const int I    = option["interpolation"].get<int>();

  engine::VelocityEngine<InternalData> velocity;
  velocity(V, D, O, P, I, get_internal_data(), delt);
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  auto      mode = option["vectorization"]["current"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;

  engine::CurrentEngine<InternalData> current;
  current(V, D, O, get_internal_data(), delt);
}

DEFINE_MEMBER(void, deposit_moment)()
{
  auto      mode = option["vectorization"]["moment"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;

  engine::MomentEngine<InternalData> moment;
  moment(V, D, O, get_internal_data());
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
