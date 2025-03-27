// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

// template trick to set number of neighbors
template <int N>
struct NbSize;

template <>
struct NbSize<1> {
  static constexpr int size = 3;
};
template <>
struct NbSize<2> {
  static constexpr int size = 9;
};
template <>
struct NbSize<3> {
  static constexpr int size = 27;
};

///
/// @brief Base class for Chunk
/// @tparam Ndim number of dimension
///
template <int Ndim>
class Chunk
{
protected:
  static constexpr int nbsize = NbSize<Ndim>::size; ///< number of neighbors

  int                  myid;           ///< chunk ID
  int                  nbid[nbsize];   ///< neighboring chunk ID
  int                  nbrank[nbsize]; ///< neighboring chunk MPI rank
  int                  dims[Ndim];     ///< number of grids
  std::vector<float64> load;           ///< load array of chunk

  virtual void initialize(const int dims[Ndim], int id)
  {
    // set ID
    set_id(id);

    // set dimensions
    for (int i = 0; i < Ndim; i++) {
      this->dims[i] = dims[i];
    }

    load.resize(1);
    reset_load();
  }

public:
  ///
  /// @brief get maximum Chunk ID allowable
  /// @return maximum Chunk ID
  ///
  static int get_max_id()
  {
    int max_int32_t = std::numeric_limits<int32_t>::max();
    return max_int32_t;
  }

  /// @brief default constructor
  Chunk()
  {
    const int dims[Ndim] = {};
    initialize(dims, 0);
  }

  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk(const int dims[Ndim], int id = 0)
  {
    initialize(dims, id);
  }

  ///
  /// @brief reset load of chunk
  ///
  virtual void reset_load()
  {
    load.assign(load.size(), 0.0);
  }

  ///
  /// @brief return load array with each element representing different types of operation
  /// @return load array
  ///
  virtual std::vector<float64> get_load()
  {
    return load;
  }

  ///
  /// @brief return sum of loads for different operations
  /// @return total load of Chunk
  ///
  virtual float64 get_total_load()
  {
    return std::accumulate(load.begin(), load.end(), 0.0);
  }

  ///
  /// @brief pack the content of Chunk into given `buffer`
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(buffer, &myid, sizeof(int), count, 0);
    count += memcpy_count(buffer, &nbid[0], nbsize * sizeof(int), count, 0);
    count += memcpy_count(buffer, &nbrank[0], nbsize * sizeof(int), count, 0);

    // load
    {
      int size = load.size();
      count += memcpy_count(buffer, &size, sizeof(int), count, 0);
      count += memcpy_count(buffer, load.data(), sizeof(float64) * size, count, 0);
    }

    return count;
  }

  ///
  /// @brief unpack the content of Chunk from given `buffer`
  /// @param buffer point to buffer from unpack
  /// @param address first address of buffer from which the data will be unpacked
  /// @return `address` + (number of bytes unpacked as a result)
  ///
  virtual int unpack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(&myid, buffer, sizeof(int), 0, count);
    count += memcpy_count(&nbid[0], buffer, nbsize * sizeof(int), 0, count);
    count += memcpy_count(&nbrank[0], buffer, nbsize * sizeof(int), 0, count);

    // load
    {
      int size = 0;
      count += memcpy_count(&size, buffer, sizeof(int), 0, count);
      load.resize(size);
      count += memcpy_count(load.data(), buffer, sizeof(float64) * size, 0, count);
    }

    return count;
  }

  ///
  /// @brief query status of boundary exchange
  /// @param mode mode of boundary exchange
  /// @param sendrecv +1 for send, -1 for recv, 0 for both
  /// @return 1 if boundary exchange is finished and 0 otherwise
  ///
  virtual int set_boundary_query(int mode, int sendrecv) = 0;

  ///
  /// @brief pack boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_pack(int mode) = 0;

  ///
  /// @brief unpack boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_unpack(int mode) = 0;

  ///
  /// @brief begin boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_begin(int mode) = 0;

  ///
  /// @brief end boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_end(int mode) = 0;

  ///
  /// @brief set Chunk ID
  /// @param id ID to be set
  ///
  void set_id(int id)
  {
    myid = id;
  }

  ///
  /// @brief get Chunk ID
  /// @return Chunk ID
  ///
  int get_id()
  {
    return myid;
  }

  ///
  /// @brief set neighbor ID for 1D Chunk
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(int dirx, int id);

  ///
  /// @brief set neighbor ID for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(int diry, int dirx, int id);

  ///
  /// @brief set neighbor ID for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(int dirz, int diry, int dirx, int id);

  ///
  /// @brief get neighbor Chunk ID for 1D Chunk
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(int dirx);

  ///
  /// @brief get neighbor Chunk ID for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(int diry, int dirx);

  ///
  /// @brief get neighbor Chunk ID for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(int dirz, int diry, int dirx);

  ///
  /// @brief set neighbor Chunk rank for 1D Chunk
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(int dirx, int rank);

  ///
  /// @brief set neighbor Chunk rank for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(int diry, int dirx, int rank);

  ///
  /// @brief set neigbhro Chunk rank for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(int dirz, int diry, int dirx, int rank);

  ///
  /// @brief get neighbor Chunk rank for 1D Chunk
  /// @param dirx direction x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(int dirx);

  ///
  /// @brief get neighbor Chunk rank for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(int diry, int dirx);

  ///
  /// @brief get neighbor Chunk rank for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(int dirz, int diry, int dirx);

  ///
  /// @brief get send tag for 1D Chunk
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(int dirx);

  ///
  /// @brief get send tag for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(int diry, int dirx);

  ///
  /// @brief get send tag for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(int dirz, int diry, int dirx);

  ///
  /// @brief get receive tag for 1D Chunk
  /// @param dirx direction in x
  /// @return receive tag
  ///
  int get_rcvtag(int dirx);

  ///
  /// @brief get receive tag for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return receive tag
  ///
  int get_rcvtag(int diry, int dirx);

  ///
  /// @brief get receive tag for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return receive tag
  ///
  int get_rcvtag(int dirz, int diry, int dirx);
};

template <int Ndim>
constexpr int Chunk<Ndim>::nbsize;

//
// implementation of small methods follows
//

template <>
inline void Chunk<1>::set_nb_id(int dirx, int id)
{
  nbid[(dirx + 1)] = id;
}

template <>
inline void Chunk<2>::set_nb_id(int diry, int dirx, int id)
{
  nbid[3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline void Chunk<3>::set_nb_id(int dirz, int diry, int dirx, int id)
{
  nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline int Chunk<1>::get_nb_id(int dirx)
{
  return nbid[(dirx + 1)];
}

template <>
inline int Chunk<2>::get_nb_id(int diry, int dirx)
{
  return nbid[3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<3>::get_nb_id(int dirz, int diry, int dirx)
{
  return nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
}

template <>
inline void Chunk<1>::set_nb_rank(int dirx, int rank)
{
  nbrank[(dirx + 1)] = rank;
}

template <>
inline void Chunk<2>::set_nb_rank(int diry, int dirx, int rank)
{
  nbrank[3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline void Chunk<3>::set_nb_rank(int dirz, int diry, int dirx, int rank)
{
  nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline int Chunk<1>::get_nb_rank(int dirx)
{
  return nbrank[(dirx + 1)];
}

template <>
inline int Chunk<2>::get_nb_rank(int diry, int dirx)
{
  return nbrank[3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<3>::get_nb_rank(int dirz, int diry, int dirx)
{
  return nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<1>::get_sndtag(int dirx)
{
  int dir = (dirx + 1);
  // return dummy tag for invalid neighbor
  return nbid[dir] < 0 ? myid : nbid[dir] % MAX_CHUNK_PER_RANK;
}

template <>
inline int Chunk<2>::get_sndtag(int diry, int dirx)
{
  int dir = 3 * (diry + 1) + (dirx + 1);
  // return dummy tag for invalid neighbor
  return nbid[dir] < 0 ? myid : nbid[dir] % MAX_CHUNK_PER_RANK;
}

template <>
inline int Chunk<3>::get_sndtag(int dirz, int diry, int dirx)
{
  int dir = 9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1);
  // return dummy tag for invalid neighbor
  return nbid[dir] < 0 ? myid : nbid[dir] % MAX_CHUNK_PER_RANK;
}

template <>
inline int Chunk<1>::get_rcvtag(int dirx)
{
  return myid % MAX_CHUNK_PER_RANK;
}

template <>
inline int Chunk<2>::get_rcvtag(int diry, int dirx)
{
  return myid % MAX_CHUNK_PER_RANK;
}

template <>
inline int Chunk<3>::get_rcvtag(int dirz, int diry, int dirx)
{
  return myid % MAX_CHUNK_PER_RANK;
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
