// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Base class for 3D Chunk
///
class Chunk3D : public Chunk<3>
{
public:
  using IntArray = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using Comm     = xt::xtensor_fixed<MPI_Comm, xt::xshape<3, 3, 3>>;
  using Request  = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;
  using Datatype = xt::xtensor_fixed<MPI_Datatype, xt::xshape<3, 3, 3>>;

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

  int get_boundary_margin() const
  {
    return boundary_margin;
  }

  ///
  /// @brief return if x dimension is ignorable
  ///
  bool has_xdim() const
  {
    return has_dim[2];
  }

  ///
  /// @brief return if y dimension is ignorable
  ///
  bool has_ydim() const
  {
    return has_dim[1];
  }

  ///
  /// @brief return if z dimension is ignorable
  ///
  bool has_zdim() const
  {
    return has_dim[0];
  }

  float64 get_delx() const
  {
    return delx;
  }

  float64 get_dely() const
  {
    return dely;
  }

  float64 get_delz() const
  {
    return delz;
  }

  auto get_xbound() const
  {
    return std::pair(Lbx, Ubx);
  }

  auto get_ybound() const
  {
    return std::pair(Lby, Uby);
  }

  auto get_zbound() const
  {
    return std::pair(Lbz, Ubz);
  }

  auto get_xrange() const
  {
    float64 xmin = -std::numeric_limits<float64>::max();
    float64 xmax = +std::numeric_limits<float64>::max();

    if (has_xdim()) {
      xmin = xlim[0];
      xmax = xlim[1];
    }

    return std::pair(xmin, xmax);
  }

  auto get_yrange() const
  {
    float64 ymin = -std::numeric_limits<float64>::max();
    float64 ymax = +std::numeric_limits<float64>::max();

    if (has_ydim()) {
      ymin = ylim[0];
      ymax = ylim[1];
    }

    return std::pair(ymin, ymax);
  }

  auto get_zrange() const
  {
    float64 zmin = -std::numeric_limits<float64>::max();
    float64 zmax = +std::numeric_limits<float64>::max();

    if (has_zdim()) {
      zmin = zlim[0];
      zmax = zlim[1];
    }

    return std::pair(zmin, zmax);
  }

  auto get_xrange_global() const
  {
    return std::pair(gxlim[0], gxlim[1]);
  }

  auto get_yrange_global() const
  {
    return std::pair(gylim[0], gylim[1]);
  }

  auto get_zrange_global() const
  {
    return std::pair(gzlim[0], gzlim[1]);
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
  void pack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
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

  ///
  /// @brief unpack for boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void unpack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
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

  ///
  /// @brief pack and start boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void begin_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
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

  ///
  /// @brief wait boundary exchange and unpack
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void end_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
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
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
