// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "nix.hpp"
#include "particle.hpp"
#include "primitives.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

using primitives::digitize;

///
/// @brief Base class for 3D Chunk
///
template <typename Particle>
class Chunk3D : public Chunk<3>
{
public:
  using IntArray    = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using Comm        = xt::xtensor_fixed<MPI_Comm, xt::xshape<3, 3, 3>>;
  using Request     = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;
  using Datatype    = xt::xtensor_fixed<MPI_Datatype, xt::xshape<3, 3, 3>>;
  using ParticlePtr = std::shared_ptr<Particle>;
  using ParticleVec = std::vector<ParticlePtr>;

  ///
  /// @brief MPI buffer
  ///
  struct MpiBuffer {
    bool     sendwait;
    bool     recvwait;
    Buffer   sendbuf;
    Buffer   recvbuf;
    IntArray bufsize;
    IntArray bufaddr;
    Comm     comm;
    Request  sendreq;
    Request  recvreq;
    Datatype sendtype;
    Datatype recvtype;

    ///
    /// constructor
    ///
    MpiBuffer() : sendwait(false), recvwait(false)
    {
    }

    ///
    /// @brief get size of buffer in bytes
    /// @return size in bytes
    ///
    int64_t get_size_byte() const
    {
      int64_t size = 0;
      size += sizeof(sendwait);
      size += sizeof(recvwait);
      size += sendbuf.size;
      size += recvbuf.size;
      size += bufsize.size() * sizeof(int);
      size += bufaddr.size() * sizeof(int);
      size += comm.size() * sizeof(MPI_Comm);
      size += sendreq.size() * sizeof(MPI_Request);
      size += recvreq.size() * sizeof(MPI_Request);
      size += sendtype.size() * sizeof(MPI_Datatype);
      size += recvtype.size() * sizeof(MPI_Datatype);
      return size;
    }

    ///
    /// @brief return send buffer for given direction
    /// @param iz z direction index
    /// @param iy y direction index
    /// @param ix x direction index
    /// @return buffer pointer
    ///
    void* get_send_buffer(int iz, int iy, int ix)
    {
      return sendbuf.get(bufaddr(iz, iy, ix));
    }

    ///
    /// @brief return recv buffer for given direction
    /// @param iz z direction index
    /// @param iy y direction index
    /// @param ix x direction index
    /// @return buffer pointer
    ///
    void* get_recv_buffer(int iz, int iy, int ix)
    {
      return recvbuf.get(bufaddr(iz, iy, ix));
    }

    ///
    /// @brief pack the content into given `buffer`
    /// @param buffer pointer to buffer to pack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int pack(void* buffer, int address)
    {
      int count = address;
      int ssize = sendbuf.size;
      int rsize = recvbuf.size;
      int asize = bufsize.size() * sizeof(int);

      count += memcpy_count(buffer, &sendwait, sizeof(bool), count, 0);
      count += memcpy_count(buffer, &recvwait, sizeof(bool), count, 0);
      count += memcpy_count(buffer, &ssize, sizeof(int), count, 0);
      count += memcpy_count(buffer, &rsize, sizeof(int), count, 0);
      count += memcpy_count(buffer, bufsize.data(), asize, count, 0);
      count += memcpy_count(buffer, bufaddr.data(), asize, count, 0);

      return count;
    }

    ///
    /// @brief unpack the content from given `buffer`
    /// @param buffer pointer to buffer from unpack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int unpack(void* buffer, int address)
    {
      int count = address;
      int ssize = 0;
      int rsize = 0;
      int asize = bufsize.size() * sizeof(int);

      count += memcpy_count(&sendwait, buffer, sizeof(bool), 0, count);
      count += memcpy_count(&recvwait, buffer, sizeof(bool), 0, count);
      count += memcpy_count(&ssize, buffer, sizeof(int), 0, count);
      count += memcpy_count(&rsize, buffer, sizeof(int), 0, count);
      count += memcpy_count(bufsize.data(), buffer, asize, 0, count);
      count += memcpy_count(bufaddr.data(), buffer, asize, 0, count);

      // memory allocation
      sendbuf.resize(ssize);
      recvbuf.resize(rsize);

      return count;
    }
  };
  using MpiBufferPtr = std::shared_ptr<MpiBuffer>;
  using MpiBufferVec = std::vector<MpiBufferPtr>;

protected:
  bool has_dim[3]; ///< flag to indicate if the dimension is ignorable

  int boundary_margin; ///< boundary margin
  int gdims[3];        ///< global number of grids
  int offset[3];       ///< global index offset
  int Lbx;             ///< lower bound in x
  int Ubx;             ///< upper bound in x
  int Lby;             ///< lower bound in y
  int Uby;             ///< upper bound in y
  int Lbz;             ///< lower bound in z
  int Ubz;             ///< upper bound in z
  int indexlb[3];      ///< index lower bound for MPI data exchange
  int indexub[3];      ///< index upper bound for MPI data exchange
  int dirlb[3];        ///< direction lower bound for MPI data exchange
  int dirub[3];        ///< direction upper bound for MPI data exchange
  int sendlb[3][3];    ///< lower bound for send
  int sendub[3][3];    ///< upper bound for send
  int recvlb[3][3];    ///< lower bound for recv
  int recvub[3][3];    ///< upper bound for recv

  float64      delx;      ///< grid size in x
  float64      dely;      ///< grid size in y
  float64      delz;      ///< grid size in z
  float64      xlim[3];   ///< physical domain in x
  float64      ylim[3];   ///< physical domain in y
  float64      zlim[3];   ///< physical domain in z
  float64      gxlim[3];  ///< global physical domain in x
  float64      gylim[3];  ///< global physical domain in y
  float64      gzlim[3];  ///< global physical domain in z
  MpiBufferVec mpibufvec; ///< MPI buffer vector
  json         option;    ///< internal option

public:
  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk3D(const int dims[3], const bool has_dim[3], int id = 0);

  ///
  /// @brief set boundary margin
  /// @param margin boundary margin
  ///
  void set_boundary_margin(int margin);

  ///
  /// @brief setup initial condition (pure virtual)
  /// @param config configuration
  ///
  virtual void setup(json& config) = 0;

  ///
  /// @brief probe incoming messages and call recv if ready
  /// @param mode mode of boundary exchange
  /// @param wait blocking if true and non-blocking otherwise
  ///
  virtual bool set_boundary_probe(int mode, bool wait) = 0;

  ///
  /// @brief pack for boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_pack(int mode) override = 0;

  ///
  /// @brief unpack for boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_unpack(int mode) override = 0;

  ///
  /// @brief begin boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_begin(int mode) override = 0;

  ///
  /// @brief end boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_end(int mode) override = 0;

  ///
  /// @brief return (approximate) size of Chunk in byte
  /// @return size of Chunk in byte
  ///
  virtual int64_t get_size_byte() = 0;

  ///
  /// @brief pack the content of Chunk into given `buffer`
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack(void* buffer, int address) override;

  ///
  /// @brief unpack the content of Chunk from given `buffer`
  /// @param buffer point to buffer from unpack
  /// @param address first address of buffer from which the data will be unpacked
  /// @return `address` + (number of bytes unpacked as a result)
  ///
  virtual int unpack(void* buffer, int address) override;

  ///
  /// @brief set coordinate of Chunk (using gdims and offset)
  /// @param dz grid size in z direction
  /// @param dy grid size in y direction
  /// @param dx grid size in x direction
  ///
  virtual void set_coordinate(float64 dz, float64 dy, float64 dx);

  ///
  /// @brief set the global context of Chunk
  /// @param offset offset for each direction in global dimensions
  /// @param gdims global number of grids for each direction
  ///
  virtual void set_global_context(const int* offset, const int* gdims);

  ///
  /// @brief set MPI communicator to MpiBuffer of given `mode`
  /// @param mode mode to specify MpiBuffer
  /// @param comm MPI communicator to be set to MpiBuffer
  ///
  virtual void set_mpi_communicator(int mode, int iz, int iy, int ix, MPI_Comm& comm);

  ///
  /// @brief count particles in cells to prepare for sorting
  /// @param particle particle species
  /// @param Lbp first index of particle array to be counted
  /// @param Ubp last index of particle array to be counted (inclusive)
  /// @param reset reset the count array before counting
  ///
  virtual void count_particle(ParticlePtr particle, int Lbp, int Ubp, bool reset = true);

  ///
  /// @brief perform particle sorting
  /// @param particle list of particle to be sorted
  ///
  virtual void sort_particle(ParticleVec& particle);

  ///
  /// @brief inject particle into the system
  /// @param particle list of particle to which new particles will be injected
  ///
  virtual void inject_particle(ParticleVec& particle);

  ///
  /// @brief query status of boundary exchange
  /// @param mode mode of boundary exchange
  /// @param sendrecv +1 for send, -1 for recv, 0 for both
  /// @return 1 if boundary exchange is finished and 0 otherwise
  ///
  virtual int set_boundary_query(int mode = 0, int sendrecv = 0) override;

  ///
  /// @brief set field boundary condition
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_field(int mode = 0);

  ///
  /// @brief set boundary condition to particle array
  /// @param particle particle species
  /// @param Lbp first index of particle array
  /// @param Ubp last index of particle array (inclusive)
  ///
  virtual void set_boundary_particle(ParticlePtr particle, int Lbp, int Ubp, int species);

  ///
  /// @brief set boundary condition to particle array after MPI send/recv
  /// @param particle particle species
  /// @param Lbp first index of particle array
  /// @param Ubp last index of particle array (inclusive)
  ///
  virtual void set_boundary_particle_after_sendrecv(ParticlePtr particle, int Lbp, int Ubp,
                                                    int species);

  ///
  /// @brief setup MPI Buffer
  /// @param mpibuf MPI buffer to be setup
  /// @param mode +1 for send, -1 for recv, 0 for both
  /// @param headbyte number of bytes used for header
  /// @param elembyte number of bytes for each element
  ///
  void set_mpi_buffer(MpiBufferPtr mpibuf, int mode, int headbyte, int elembyte);

  ///
  /// @brief return MpiBuffer of given mode of boundary exchange
  /// @param mode mode of MpiBuffer
  /// @return MpiBufferPtr or std::shared_ptr<MpiBuffer>
  ///
  MpiBufferPtr get_mpi_buffer(int mode)
  {
    return mpibufvec[mode];
  }

  ///
  /// @brief get buffer ratio (relative to the required size) from configuration file
  /// @return buffer ratio
  ///
  float64 get_buffer_ratio()
  {
    return option.value("buffer_ratio", 0.2);
  }

  ///
  /// @brief return if x dimension is ignorable
  ///
  bool has_x_dim() const
  {
    return has_dim[2];
  }

  ///
  /// @brief return if y dimension is ignorable
  ///
  bool has_y_dim() const
  {
    return has_dim[1];
  }

  ///
  /// @brief return if z dimension is ignorable
  ///
  bool has_z_dim() const
  {
    return has_dim[0];
  }

  ///
  /// @brief get lower bound in x
  ///
  float64 get_xmin()
  {
    if (has_x_dim()) {
      return xlim[0];
    } else {
      return -std::numeric_limits<float64>::max();
    }
  }

  ///
  /// @brief get upper bound in x
  ///
  float64 get_xmax()
  {
    if (has_x_dim()) {
      return xlim[1];
    } else {
      return +std::numeric_limits<float64>::max();
    }
  }

  ///
  /// @brief get lower bound in y
  ///
  float64 get_ymin()
  {
    if (has_y_dim()) {
      return ylim[0];
    } else {
      return -std::numeric_limits<float64>::max();
    }
  }

  ///
  /// @brief get upper bound in y
  ///
  float64 get_ymax()
  {
    if (has_y_dim()) {
      return ylim[1];
    } else {
      return +std::numeric_limits<float64>::max();
    }
  }

  ///
  /// @brief get lower bound in z
  ///
  float64 get_zmin()
  {
    if (has_z_dim()) {
      return zlim[0];
    } else {
      return -std::numeric_limits<float64>::max();
    }
  }

  ///
  /// @brief get upper bound in z
  ///
  float64 get_zmax()
  {
    if (has_z_dim()) {
      return zlim[1];
    } else {
      return +std::numeric_limits<float64>::max();
    }
  }

protected:
  ///
  /// @brief probe incoming messages
  /// @param mpibuf MPI buffer
  /// @return true 1 recv has been called and 0 otherwise
  ///
  int probe_bc_exchange(MpiBufferPtr mpibuf);

  ///
  /// @brief pack for boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void pack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo);

  ///
  /// @brief unpack for boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void unpack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo);

  ///
  /// @brief pack and start boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void begin_bc_exchange(MpiBufferPtr mpibuf, Halo& halo);

  ///
  /// @brief wait boundary exchange and unpack
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void end_bc_exchange(MpiBufferPtr mpibuf, Halo& halo);
};

//
// implementation follows
//
#define DEFINE_MEMBER(type, name)                                                                  \
  template <typename ParticlePtr>                                                                  \
  type Chunk3D<ParticlePtr>::name

DEFINE_MEMBER(, Chunk3D)
(const int dims[3], const bool has_dim[3], int id)
    : Chunk<3>(dims, id), delx(1.0), dely(1.0), delz(1.0), option({})
{
  for (int i = 0; i < 3; i++) {
    this->has_dim[i] = has_dim[i];
  }
  reset_load();
}

DEFINE_MEMBER(void, set_boundary_margin)(int margin)
{
  boundary_margin = margin;

  // direction indices
  indexlb[0] = 0;
  indexub[0] = 2;
  dirlb[0]   = -1;
  dirub[0]   = +1;
  indexlb[1] = 0;
  indexub[1] = 2;
  dirlb[1]   = -1;
  dirub[1]   = +1;
  indexlb[2] = 0;
  indexub[2] = 2;
  dirlb[2]   = -1;
  dirub[2]   = +1;

  // lower and upper bound
  Lbz = boundary_margin;
  Ubz = boundary_margin + this->dims[0] - 1;
  Lby = boundary_margin;
  Uby = boundary_margin + this->dims[1] - 1;
  Lbx = boundary_margin;
  Ubx = boundary_margin + this->dims[2] - 1;

  if (has_z_dim() == false) {
    // z direction is ignorable coordinate
    Lbz        = boundary_margin;
    Ubz        = boundary_margin;
    indexlb[0] = 1;
    indexub[0] = 1;
    dirlb[0]   = 0;
    dirub[0]   = 0;
  }

  if (has_y_dim() == false) {
    // y direction is ignorable coordinate
    Lby        = boundary_margin;
    Uby        = boundary_margin;
    indexlb[1] = 1;
    indexub[1] = 1;
    dirlb[1]   = 0;
    dirub[1]   = 0;
  }

  if (has_x_dim() == false) {
    // x direction is ignorable coordinate
    Lbx        = boundary_margin;
    Ubx        = boundary_margin;
    indexlb[2] = 1;
    indexub[2] = 1;
    dirlb[2]   = 0;
    dirub[2]   = 0;
  }

  // MPI send
  sendlb[0][0] = Lbz;
  sendlb[0][1] = Lbz;
  sendlb[0][2] = Ubz - boundary_margin + 1;
  sendub[0][0] = Lbz + boundary_margin - 1;
  sendub[0][1] = Ubz;
  sendub[0][2] = Ubz;
  sendlb[1][0] = Lby;
  sendlb[1][1] = Lby;
  sendlb[1][2] = Uby - boundary_margin + 1;
  sendub[1][0] = Lby + boundary_margin - 1;
  sendub[1][1] = Uby;
  sendub[1][2] = Uby;
  sendlb[2][0] = Lbx;
  sendlb[2][1] = Lbx;
  sendlb[2][2] = Ubx - boundary_margin + 1;
  sendub[2][0] = Lbx + boundary_margin - 1;
  sendub[2][1] = Ubx;
  sendub[2][2] = Ubx;

  // MPI recv
  recvlb[0][0] = Lbz - boundary_margin;
  recvlb[0][1] = Lbz;
  recvlb[0][2] = Ubz + 1;
  recvub[0][0] = Lbz - 1;
  recvub[0][1] = Ubz;
  recvub[0][2] = Ubz + boundary_margin;
  recvlb[1][0] = Lby - boundary_margin;
  recvlb[1][1] = Lby;
  recvlb[1][2] = Uby + 1;
  recvub[1][0] = Lby - 1;
  recvub[1][1] = Uby;
  recvub[1][2] = Uby + boundary_margin;
  recvlb[2][0] = Lbx - boundary_margin;
  recvlb[2][1] = Lbx;
  recvlb[2][2] = Ubx + 1;
  recvub[2][0] = Lbx - 1;
  recvub[2][1] = Ubx;
  recvub[2][2] = Ubx + boundary_margin;
}

DEFINE_MEMBER(int, pack)(void* buffer, int address)
{
  int count = address;

  count = Chunk<3>::pack(buffer, count);

  // option
  {
    std::vector<uint8_t> msgpack = json::to_msgpack(option);
    int                  size    = msgpack.size();
    count += memcpy_count(buffer, &size, sizeof(int), count, 0);
    count += memcpy_count(buffer, msgpack.data(), size, count, 0);
  }

  count += memcpy_count(buffer, &boundary_margin, sizeof(int), count, 0);
  count += memcpy_count(buffer, &delx, sizeof(float64), count, 0);
  count += memcpy_count(buffer, &dely, sizeof(float64), count, 0);
  count += memcpy_count(buffer, &delz, sizeof(float64), count, 0);
  count += memcpy_count(buffer, xlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, ylim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, zlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gxlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gylim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gzlim, 3 * sizeof(float64), count, 0);
  // MPI buffer (NOTE: MPI communicator is NOT packed)
  {
    int nmode = mpibufvec.size();
    count += memcpy_count(buffer, &nmode, sizeof(int), count, 0);

    for (int mode = 0; mode < nmode; mode++) {
      count = mpibufvec[mode]->pack(buffer, count);
    }
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void* buffer, int address)
{
  int count = address;

  count = Chunk<3>::unpack(buffer, count);

  // option
  {
    int size = 0;
    count += memcpy_count(&size, buffer, sizeof(int), 0, count);

    std::vector<uint8_t> msgpack(size);
    count += memcpy_count(msgpack.data(), buffer, size, 0, count);

    option = json::from_msgpack(msgpack);
  }

  count += memcpy_count(&boundary_margin, buffer, sizeof(int), 0, count);
  count += memcpy_count(&delx, buffer, sizeof(float64), 0, count);
  count += memcpy_count(&dely, buffer, sizeof(float64), 0, count);
  count += memcpy_count(&delz, buffer, sizeof(float64), 0, count);
  count += memcpy_count(xlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(ylim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(zlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gxlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gylim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gzlim, buffer, 3 * sizeof(float64), 0, count);
  // MPI buffer (NOTE: MPI communicator is NOT unpacked)
  {
    int nmode = 0;
    count += memcpy_count(&nmode, buffer, sizeof(int), 0, count);
    mpibufvec.resize(nmode);

    for (int mode = 0; mode < nmode; mode++) {
      mpibufvec[mode] = std::make_shared<MpiBuffer>();
      count           = mpibufvec[mode]->unpack(buffer, count);
    }
  }

  // set boundary margin related variables
  set_boundary_margin(boundary_margin);

  return count;
}

DEFINE_MEMBER(void, set_coordinate)(float64 dz, float64 dy, float64 dx)
{
  // set internal data members
  delz = dz;
  dely = dy;
  delx = dx;

  // z direction
  zlim[0]  = offset[0] * delz;
  zlim[1]  = offset[0] * delz + dims[0] * delz;
  zlim[2]  = zlim[1] - zlim[0];
  gzlim[0] = 0.0;
  gzlim[1] = gdims[0] * delz;
  gzlim[2] = gzlim[1] - gzlim[0];

  // y direction
  ylim[0]  = offset[1] * dely;
  ylim[1]  = offset[1] * dely + dims[1] * dely;
  ylim[2]  = ylim[1] - ylim[0];
  gylim[0] = 0.0;
  gylim[1] = gdims[1] * dely;
  gylim[2] = gylim[1] - gylim[0];

  // x direction
  xlim[0]  = offset[2] * delx;
  xlim[1]  = offset[2] * delx + dims[2] * delx;
  xlim[2]  = xlim[1] - xlim[0];
  gxlim[0] = 0.0;
  gxlim[1] = gdims[2] * delx;
  gxlim[2] = gxlim[1] - gxlim[0];
}

DEFINE_MEMBER(void, set_global_context)(const int* offset, const int* gdims)
{
  this->gdims[0]  = gdims[0];
  this->gdims[1]  = gdims[1];
  this->gdims[2]  = gdims[2];
  this->offset[0] = offset[0];
  this->offset[1] = offset[1];
  this->offset[2] = offset[2];
}

DEFINE_MEMBER(void, set_mpi_communicator)(int mode, int iz, int iy, int ix, MPI_Comm& comm)
{
  if (mode >= 0 && mode < mpibufvec.size()) {
    mpibufvec[mode]->comm(iz, iy, ix) = comm;
  } else {
    ERROR << tfm::format("invalid index %d for mpibufvec", mode);
  }
}

DEFINE_MEMBER(void, count_particle)(ParticlePtr particle, int Lbp, int Ubp, bool reset)
{
}

DEFINE_MEMBER(void, sort_particle)(ParticleVec& particle)
{
}

DEFINE_MEMBER(void, inject_particle)(ParticleVec& particle)
{
}

DEFINE_MEMBER(int, set_boundary_query)(int mode, int sendrecv)
{
  int flag = 0;

  // MPI buffer
  MpiBufferPtr mpibuf = mpibufvec[mode];

  OMP_MAYBE_CRITICAL
  if (sendrecv == 0) {
    // both send/recv
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (sendrecv == +1) {
    // send
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (sendrecv == -1) {
    // recv
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  }

  return flag;
}

DEFINE_MEMBER(void, set_boundary_field)(int mode)
{
  // lower boundary in z
  if (get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in z
  if (get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  // lower boundary in y
  if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in y
  if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  // lower boundary in x
  if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in x
  if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
    ERROR << tfm::format("Non-periodic boundary condition has not been implemented!");
  }
}

DEFINE_MEMBER(void, set_boundary_particle)(ParticlePtr particle, int Lbp, int Ubp, int species)
{
}

DEFINE_MEMBER(void, set_boundary_particle_after_sendrecv)
(ParticlePtr particle, int Lbp, int Ubp, int species)
{
  // trick to take care of round-off error and ignorable coordinates
  const int     xdim = has_x_dim();
  const int     ydim = has_y_dim();
  const int     zdim = has_z_dim();
  const float64 xmin = gxlim[0];
  const float64 xmax = gxlim[1];
  const float64 ymin = gylim[0];
  const float64 ymax = gylim[1];
  const float64 zmin = gzlim[0];
  const float64 zmax = gzlim[1];
  const float64 xlen = xdim * (gxlim[2] - std::numeric_limits<float64>::epsilon());
  const float64 ylen = ydim * (gylim[2] - std::numeric_limits<float64>::epsilon());
  const float64 zlen = zdim * (gzlim[2] - std::numeric_limits<float64>::epsilon());

  // apply periodic boundary condition
  auto& xu = particle->xu;
  for (int ip = Lbp; ip <= Ubp; ip++) {
    xu(ip, 0) += (xu(ip, 0) < xmin) * xlen - (xu(ip, 0) >= xmax) * xlen;
    xu(ip, 1) += (xu(ip, 1) < ymin) * ylen - (xu(ip, 1) >= ymax) * ylen;
    xu(ip, 2) += (xu(ip, 2) < zmin) * zlen - (xu(ip, 2) >= zmax) * zlen;
  }
}

DEFINE_MEMBER(void, set_mpi_buffer)
(MpiBufferPtr mpibuf, int mode, int headbyte, int elembyte)
{
  int size = 0;

  for (int iz = 0; iz <= 2; iz++) {
    for (int iy = 0; iy <= 2; iy++) {
      for (int ix = 0; ix <= 2; ix++) {
        if (iz == 1 && iy == 1 && ix == 1) {
          mpibuf->bufsize(iz, iy, ix) = 0;
          mpibuf->bufaddr(iz, iy, ix) = size;
        } else {
          int nz = recvub[0][iz] - recvlb[0][iz] + 1;
          int ny = recvub[1][iy] - recvlb[1][iy] + 1;
          int nx = recvub[2][ix] - recvlb[2][ix] + 1;

          mpibuf->bufsize(iz, iy, ix) = headbyte + elembyte * nz * ny * nx;
          mpibuf->bufaddr(iz, iy, ix) = size;
          size += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
  }

  // buffer allocation
  if (mode == +1 || mode == 0) {
    mpibuf->sendbuf.resize(size);
  }
  if (mode == -1 || mode == 0) {
    mpibuf->recvbuf.resize(size);
  }
}

DEFINE_MEMBER(int, probe_bc_exchange)
(MpiBufferPtr mpibuf)
{
  // return if recv has already been called
  if (mpibuf->recvwait == true)
    return 1;

  //
  // probe incoming messages
  //
  bool is_everyone_ready = true;

  mpibuf->bufsize.fill(0);
  mpibuf->bufaddr.fill(0);

  OMP_MAYBE_CRITICAL
  for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
    for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
      for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
        // skip
        if (iz == 1 && iy == 1 && ix == 1)
          continue;

        MPI_Status status;
        int        is_ready = 0;

        auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
        auto& recvtype = mpibuf->recvtype(iz, iy, ix);
        int   nbrank   = get_nb_rank(dirz, diry, dirx);
        int   recvtag  = get_rcvtag(dirz, diry, dirx);

        MPI_Iprobe(nbrank, recvtag, recvcomm, &is_ready, &status);

        // get message size
        if (is_ready) {
          int count    = 0;
          int typebyte = 0;
          MPI_Get_count(&status, recvtype, &count);
          MPI_Type_size(recvtype, &typebyte);
          mpibuf->bufsize(iz, iy, ix) = count * typebyte;
        } else {
          // not ready yet
          is_everyone_ready = false;
        }
      }
    }
  }

  if (is_everyone_ready == false)
    return 0;

  //
  // recv incoming messages
  //
  {
    int bufsize = 0;

    // prepare for recv
    for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
      for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
        for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->bufaddr(iz, iy, ix) = bufsize;
          bufsize += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
    mpibuf->recvbuf.resize(bufsize);

    // recv
    OMP_MAYBE_CRITICAL
    for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
      for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
        for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
          // skip
          if (iz == 1 && iy == 1 && ix == 1)
            continue;

          auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
          auto& recvtype = mpibuf->recvtype(iz, iy, ix);
          auto& recvreq  = mpibuf->recvreq(iz, iy, ix);
          int   nbrank   = get_nb_rank(dirz, diry, dirx);
          int   recvtag  = get_rcvtag(dirz, diry, dirx);
          void* recvptr  = mpibuf->get_recv_buffer(iz, iy, ix);
          int   recvcnt  = mpibuf->bufsize(iz, iy, ix);

          MPI_Irecv(recvptr, recvcnt, recvtype, nbrank, recvtag, recvcomm, &recvreq);
        }
      }
    }

    mpibuf->recvwait = true;
  }

  return 1;
}

DEFINE_MEMBER(template <typename Halo> void, pack_bc_exchange)
(MpiBufferPtr mpibuf, Halo& halo)
{
  // pre-process
  halo.pre_pack(mpibuf, indexlb, indexub);

  for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
    for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
      for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
        // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
        // clang-format on

        // pack
        bool status = halo.pack(mpibuf, iz, iy, ix, send_bound, recv_bound);

        mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
        mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
      }
    }
  }

  // post-process
  halo.post_pack(mpibuf, indexlb, indexub);
}

DEFINE_MEMBER(template <typename Halo> void, unpack_bc_exchange)
(MpiBufferPtr mpibuf, Halo& halo)
{
  // pre-process
  halo.pre_unpack(mpibuf, indexlb, indexub);

  //
  // unpack recv buffer
  //
  for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
    for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
      for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
        // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
        // clang-format on

        // unpack
        bool status = halo.unpack(mpibuf, iz, iy, ix, send_bound, recv_bound);
      }
    }
  }

  // post-proces
  halo.post_unpack(mpibuf, indexlb, indexub);
}

DEFINE_MEMBER(template <typename Halo> void, begin_bc_exchange)
(MpiBufferPtr mpibuf, Halo& halo)
{
  static constexpr bool is_send_required = true;
  static constexpr bool is_recv_required = Halo::is_buffer_fixed == true;

  mpibuf->sendwait = false;
  mpibuf->recvwait = false;

  for (int iz = 0; iz <= 2; iz++) {
    for (int iy = 0; iy <= 2; iy++) {
      for (int ix = 0; ix <= 2; ix++) {
        mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
        mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
      }
    }
  }

  OMP_MAYBE_CRITICAL
  for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
    for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
      for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
        if (iz == 1 && iy == 1 && ix == 1)
          continue;

        int nbrank = get_nb_rank(dirz, diry, dirx);
        // send
        if constexpr (is_send_required == true) {
          int   sendtag  = get_sndtag(dirz, diry, dirx);
          auto& sendcomm = mpibuf->comm(1 + dirz, 1 + diry, 1 + dirx);
          auto& sendtype = mpibuf->sendtype(iz, iy, ix);
          auto& sendreq  = mpibuf->sendreq(iz, iy, ix);
          void* sendptr  = mpibuf->get_send_buffer(iz, iy, ix);
          int   sendcnt  = mpibuf->bufsize(iz, iy, ix);

          MPI_Isend(sendptr, sendcnt, sendtype, nbrank, sendtag, sendcomm, &sendreq);
        }

        // recv
        if constexpr (is_recv_required == true) {
          int   recvtag  = get_rcvtag(dirz, diry, dirx);
          auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
          auto& recvtype = mpibuf->recvtype(iz, iy, ix);
          auto& recvreq  = mpibuf->recvreq(iz, iy, ix);
          void* recvptr  = mpibuf->get_recv_buffer(iz, iy, ix);
          int   recvcnt  = mpibuf->bufsize(iz, iy, ix);

          MPI_Irecv(recvptr, recvcnt, recvtype, nbrank, recvtag, recvcomm, &recvreq);
        }
      }
    }
  }

  if constexpr (is_send_required == true) {
    mpibuf->sendwait = true;
  }

  if constexpr (is_recv_required == true) {
    mpibuf->recvwait = true;
  }
}

DEFINE_MEMBER(template <typename Halo> void, end_bc_exchange)
(MpiBufferPtr mpibuf, Halo& halo)
{
  // wait for MPI send/recv calls to complete
  OMP_MAYBE_CRITICAL
  {
    if (mpibuf->sendwait == true) {
      MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUSES_IGNORE);
      mpibuf->sendwait = false;
    }

    if (mpibuf->recvwait == true) {
      MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUSES_IGNORE);
      mpibuf->recvwait = false;
    }
  }
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
