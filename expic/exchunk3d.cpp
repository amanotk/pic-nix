// -*- C++ -*-
#include "exchunk3d.hpp"

#include "engine.hpp"

#define DEFINE_MEMBER(type, name) type ExChunk3D::name

DEFINE_MEMBER(, ExChunk3D)
(const int dims[3], int id) : Chunk(dims, id), Ns(1)
{
#if 0
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
#endif
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

  // order of shape function
  {
    order = opt.value("order", 2);
    this->set_boundary_margin((order + 3) / 2);
  }

  // vectorization mode
  {
    std::vector<std::string> valid_mode    = {"scalar", "xsimd", "xsimd-unsorted"};
    auto                     vectorization = opt["vectorization"];

    bool is_success = true;

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
                                   {"velocity", "xsimd"},
                                   {"current", "xsimd"},
                                   {"moment", "xsimd"}};
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

  // seed for random number generator
  {
    auto seed_type = opt["seed_type"];

    if (seed_type.is_string() == false || seed_type == "random") {
      option["random_seed"] = std::random_device()();
    } else if (seed_type == "fixed") {
      option["random_seed"] = this->myid; // chunk ID
    } else {
      ERROR << tfm::format("Invalid seed type: %s", seed_type);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // interpolation method
  {
    auto interp = opt["interpolation"];

    if (interp.is_string() == false || interp == "mc") {
      option["interpolation"] = "mc";
    } else if (interp == "wt") {
      option["interpolation"] = "wt";
    } else {
      ERROR << tfm::format("Invalid interpolation method: %s", interp);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Friedman filter
  {
    option["friedman"] = opt.value("friedman", 0.0);
  }

  // misc
  {
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
  std::string mode = option["vectorization"]["position"].get<std::string>();

  bool is_success = true;

  if (mode == "scalar") {
    engine::ScalarPosition position(get_internal_data());
    position(up, delt);
  } else if (mode == "xsimd") {
    engine::VectorPosition position(get_internal_data());
    position(up, delt);
  } else {
    is_success = false;
  }

  // apply boundary condition and count particles
  for (int is = 0; is < Ns; is++) {
    set_boundary_particle(up[is], 0, up[is]->Np - 1, is);
    count_particle(up[is], 0, up[is]->Np - 1, true);
  }

  if (is_success == false) {
    ERROR << tfm::format("Error detected in push_position (see below):");
    std::cerr << std::setw(2) << option << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  constexpr int Order = 2;

  bool        is_success = true;
  std::string mode       = option["vectorization"]["velocity"].get<std::string>();
  std::string interp     = option["interpolation"].get<std::string>();

  if (interp == "mc") {
    //
    // MC interpolation
    //
    if (mode == "scalar") {
      engine::ScalarVelocityBorisMC<3, Order> velocity(get_internal_data());
      velocity(up, uf, delt);
    } else if (mode == "xsimd") {
      engine::VectorVelocityBorisMC<3, Order> velocity(get_internal_data());
      velocity(up, uf, delt);
    } else {
      is_success = false;
    }
  } else if (interp == "wt") {
    //
    // WT interpolation
    //
    if (mode == "scalar") {
      engine::ScalarVelocityBorisWT<3, Order> velocity(get_internal_data());
      velocity(up, uf, delt);
    } else if (mode == "xsimd") {
      engine::VectorVelocityBorisWT<3, Order> velocity(get_internal_data());
      velocity(up, uf, delt);
    } else {
      is_success = false;
    }
  } else {
    is_success = false;
  }

  if (is_success == false) {
    ERROR << tfm::format("Error detected in push_velocity (see below):");
    std::cerr << std::setw(2) << option << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  constexpr int Order = 2;

  bool        is_success = true;
  std::string mode       = option["vectorization"]["current"].get<std::string>();

  if (mode == "scalar") {
    engine::ScalarCurrent<3, Order> current(get_internal_data());
    current(up, uj, delt);
  } else if (mode == "xsimd") {
    engine::VectorCurrent<3, Order> current(get_internal_data());
    current(up, uj, delt);
  } else {
    is_success = false;
  }

  if (is_success == false) {
    ERROR << tfm::format("Error detected in deposit_current (see below):");
    std::cerr << std::setw(2) << option << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  constexpr int Order = 2;

  bool        is_success = true;
  std::string mode       = option["vectorization"]["moment"].get<std::string>();

  if (mode == "scalar") {
    engine::ScalarMoment<3, Order> moment(get_internal_data());
    moment(up, um);
  } else if (mode == "xsimd") {
    engine::VectorMoment<3, Order> moment(get_internal_data());
    moment(up, um);
  } else {
    is_success = false;
  }

  if (is_success == false) {
    ERROR << tfm::format("Error detected in deposit_moment (see below):");
    std::cerr << std::setw(2) << option << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
