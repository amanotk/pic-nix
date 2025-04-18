// -*- C++ -*-
#include "pic_chunk.hpp"
#include "pic_engine.hpp"

PicChunk::PicChunk(const int dims[3], const bool has_dim[3], int id)
    : base_type(dims, has_dim, id), Ns(1)
{
  // check dimension
  {
    bool is_3d = has_dim[0] == true && has_dim[1] == true && has_dim[2] == true;
    bool is_2d = has_dim[0] == false && has_dim[1] == true && has_dim[2] == true;
    bool is_1d = has_dim[0] == false && has_dim[1] == false && has_dim[2] == true;

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

int64_t PicChunk::get_size_byte() const
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

int PicChunk::get_order() const
{
  return order;
}

int PicChunk::pack(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count = base_type::pack(buffer, count);

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

int PicChunk::unpack(void* buffer, int address)
{
  using nix::memcpy_count;

  int count = address;

  count = base_type::unpack(buffer, count);

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

void PicChunk::allocate()
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

void PicChunk::reset_load()
{
  const int Ng = dims[0] * dims[1] * dims[2];

  load[LoadField]    = option.value("cell_load", 1.0);
  load[LoadParticle] = 0;
  for (int is = 0; is < up.size(); is++) {
    load[LoadParticle] += up[is]->Np / Ng;
  }
}

void PicChunk::setup(json& config)
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

    int nb = (order + 3) / 2; // required boundary margin

    bool is_x_valid = has_dim[2] == false || dims[2] >= nb;
    bool is_y_valid = has_dim[1] == false || dims[1] >= nb;
    bool is_z_valid = has_dim[0] == false || dims[0] >= nb;

    if (is_x_valid == false || is_y_valid == false || is_z_valid == false) {
      ERROR << tfm::format("Number of grid points smaller than the minimum (%2d) "
                           "for the chosen shape order (%2d) of shape function ",
                           nb, order);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    option["order"] = order;
    this->set_boundary_margin(nb);
  }

  // pusher
  {
    auto pusher = opt.value("pusher", "Boris");

    if (pusher == "Boris") {
      option["pusher"] = pic_engine::PusherBoris;
    } else if (pusher == "Vay") {
      option["pusher"] = pic_engine::PusherVay;
    } else if (pusher == "HigueraCary") {
      option["pusher"] = pic_engine::PusherHigueraCary;
    } else {
      ERROR << tfm::format("Invalid pusher: %s", pusher);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // interpolation
  {
    auto interpolation = opt.value("interpolation", "MC");

    if (interpolation == pic_engine::InterpName[pic_engine::InterpMC]) {
      option["interpolation"] = pic_engine::InterpMC;
    } else if (interpolation == pic_engine::InterpName[pic_engine::InterpWT]) {
      option["interpolation"] = pic_engine::InterpWT;
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

void PicChunk::init_friedman()
{
  pic_engine::Maxwell<this_type> maxwell;
  maxwell.init_friedman(*this, get_internal_data());
}

bool PicChunk::set_boundary_probe(int mode, bool wait)
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

void PicChunk::set_boundary_pack(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<this_type>(uf, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<this_type>(uj, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<this_type>(um, *this);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<this_type>(up, *this);
    this->inject_particle(up);
    this->pack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }
}

void PicChunk::set_boundary_unpack(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<this_type>(uf, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<this_type>(uj, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<this_type>(um, *this);
    this->unpack_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<this_type>(up, *this);
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

void PicChunk::set_boundary_begin(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<this_type>(uf, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<this_type>(uj, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<this_type>(um, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<this_type>(up, *this);
    this->begin_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }
}

void PicChunk::set_boundary_end(int mode)
{
  switch (mode) {
  case BoundaryEmf: {
    auto halo = nix::XtensorHaloField3D<this_type>(uf, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryCur: {
    auto halo = nix::XtensorHaloCurrent3D<this_type>(uj, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryMom: {
    auto halo = nix::XtensorHaloMoment3D<this_type>(um, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  case BoundaryParticle: {
    auto halo = nix::XtensorHaloParticle3D<this_type>(up, *this);
    this->end_bc_exchange(mpibufvec[mode], halo);
    break;
  }
  default:
    ERROR << tfm::format("No such boundary mode exists!");
    break;
  }
}

void PicChunk::get_energy(float64& efd, float64& bfd, float64 particle[])
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

void PicChunk::get_diverror(float64& efd, float64& bfd)
{
  pic_engine::Maxwell<this_type> maxwell;
  std::tie(efd, bfd) = maxwell.get_diverror(dimension, *this, get_internal_data());
}

void PicChunk::sort_particle(ParticleVec& particle)
{
  for (int is = 0; is < particle.size(); is++) {
    particle[is]->count(0, particle[is]->Np - 1, true, order);
    particle[is]->sort();
  }
}

void PicChunk::set_boundary_field(int mode)
{

  if (get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  if (get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }
}

void PicChunk::set_boundary_particle(ParticleVec& particle)
{
}

void PicChunk::inject_particle(ParticleVec& particle)
{
}

void PicChunk::push_position(const float64 delt)
{
  auto      mode = option["vectorization"]["position"].get<std::string>();
  const int V    = "vector" == mode;
  const int O    = order;

  pic_engine::Position<this_type> position;
  position(V, O, *this, get_internal_data(), delt);
}

void PicChunk::push_velocity(const float64 delt)
{
  auto      mode = option["vectorization"]["velocity"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;
  const int P    = option["pusher"].get<int>();
  const int I    = option["interpolation"].get<int>();

  pic_engine::Velocity<this_type> velocity;
  velocity(V, D, O, P, I, *this, get_internal_data(), delt);
}

void PicChunk::deposit_current(const float64 delt)
{
  auto      mode = option["vectorization"]["current"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;

  pic_engine::Current<this_type> current;
  current(V, D, O, *this, get_internal_data(), delt);
}

void PicChunk::deposit_moment()
{
  auto      mode = option["vectorization"]["moment"].get<std::string>();
  const int V    = "vector" == mode;
  const int D    = dimension;
  const int O    = order;

  pic_engine::Moment<this_type> moment;
  moment(V, D, O, *this, get_internal_data());
}

void PicChunk::push_efd(float64 delt)
{
  const int D = dimension;

  pic_engine::Maxwell<this_type> maxwell;
  maxwell.push_efd(D, *this, get_internal_data(), delt);
}

void PicChunk::push_bfd(float64 delt)
{
  const int D = dimension;

  pic_engine::Maxwell<this_type> maxwell;
  maxwell.push_bfd(D, *this, get_internal_data(), delt);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
