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
  // config
  {
    std::vector<uint8_t> msgpack = json::to_msgpack(config);
    int                  size    = msgpack.size();
    count += memcpy_count(buffer, &size, sizeof(int), count, 0);
    count += memcpy_count(buffer, msgpack.data(), size, count, 0);
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
  // config
  {
    int size = 0;
    count += memcpy_count(&size, buffer, sizeof(int), 0, count);

    std::vector<uint8_t> msgpack(size);
    count += memcpy_count(msgpack.data(), buffer, size, 0, count);

    config = json::from_msgpack(msgpack);
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
  this->config = config;

  // vectorization mode
  {
    std::vector<std::string> valid_mode    = {"scalar", "xsimd", "xsimd-unsorted"};
    auto                     vectorization = this->config["vectorization"];

    bool is_object = vectorization.is_object() == true;
    bool is_scalar = is_object == false && vectorization == "scalar";
    bool is_vector = is_object == false && vectorization == "vector";

    if (is_scalar == true) {
      // clang-format off
      this->config["vectorization"] = {{"position", "scalar"},
                                       {"velocity", "scalar"},
                                       {"current", "scalar"},
                                       {"moment", "scalar"}};
      // clang-format on
    } else if (is_vector == true) {
      // clang-format off
      this->config["vectorization"] = {{"position", "scalar"},
                                       {"velocity", "xsimd"},
                                       {"current", "xsimd"},
                                       {"moment", "xsimd"}};
      // clang-format on
    } else if (is_object == true) {
      for (auto& key : {"position", "velocity", "current", "moment"}) {
        std::string mode = vectorization.value(key, "scalar");

        bool is_valid_mode =
            std::find(valid_mode.begin(), valid_mode.end(), mode) != valid_mode.end();
        if (is_valid_mode == true) {
          this->config["vectorization"][key] = mode;
        } else {
          ERROR << tfm::format("Invalid vectorization mode for %s: %s", key, mode);
          MPI_Abort(MPI_COMM_WORLD, -1);
        }
      }
    }

    // print vectorization mode
    for (auto& key : {"position", "velocity", "current", "moment"}) {
      std::string mode = this->config["vectorization"][key];
      DEBUG1 << tfm::format("Vectorization mode for %s: %s", key, mode);
    }
  }

  // seed for random number generator
  {
    if (this->config["seed"].is_null() == true) {
      this->config["seed"] = "random";
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

DEFINE_MEMBER(void, count_particle)(ParticlePtr particle, int Lbp, int Ubp, bool reset)
{
  // notice the half-grid offset of cell boundaries for odd-order shape functions
  constexpr int is_odd        = (Order % 2 == 1) ? 1 : 0;
  const int     out_of_bounds = particle->Ng;
  const float64 xmin          = xlim[0] - 0.5 * delx * is_odd;
  const float64 ymin          = ylim[0] - 0.5 * dely * is_odd;
  const float64 zmin          = zlim[0] - 0.5 * delz * is_odd;

  // parameters for sort by cell
  int     stride_x = 1;
  int     stride_y = stride_x * (dims[2] + 1);
  int     stride_z = stride_y * (dims[1] + 1);
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
  auto mode = config["vectorization"].value("position", "scalar");

  if (mode == "scalar") {
    push_position_impl_scalar(delt);
  } else if (mode == "xsimd") {
    push_position_impl_xsimd(delt);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  auto mode = config["vectorization"].value("velocity", "scalar");

  if (mode == "scalar") {
    push_velocity_impl_scalar(delt);
  } else if (mode == "xsimd") {
    push_velocity_impl_xsimd(delt);
  } else if (mode == "xsimd-unsorted") {
    push_velocity_unsorted_impl_xsimd(delt);
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  auto mode = config["vectorization"].value("current", "scalar");

  if (mode == "scalar") {
    deposit_current_impl_scalar(delt);
  } else if (mode == "xsimd") {
    deposit_current_impl_xsimd(delt);
  } else if (mode == "xsimd-unsorted") {
    deposit_current_unsorted_impl_xsimd(delt);
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  deposit_moment_impl_scalar();
}

#undef DEFINE_MEMBER

#include "exchunk3d_impl.hpp"
#include "exchunk3d_impl_scalar.cpp" // scalar version
#include "exchunk3d_impl_xsimd.cpp"  // vector version with xsimd

template class ExChunk3D<1>;
template class ExChunk3D<2>;
template class ExChunk3D<3>;
template class ExChunk3D<4>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
