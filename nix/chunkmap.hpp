// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

#include "nix.hpp"
#include "sfc.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief ChunkMap class
///
/// The ChunkMap class provides a bidirectional mapping between a linear chunk ID
/// used for identifying a block in a distributed system and its corresponding
/// 3D Cartesian coordinates within a grid. This mapping facilitates operations
/// such as neighbor lookup, boundary condition handling, and rank assignment
/// in parallel processing scenarios.
///
class ChunkMap
{
protected:
  using IntArray1D = xt::xtensor<int, 1>;
  using IntArray2D = xt::xtensor<int, 2>;
  using IntArray3D = xt::xtensor<int, 3>;

  int              size;           ///< number of total chunks
  int              dims[3];        ///< chunk dimension
  int              periodicity[3]; ///< periodicity in each direction
  std::vector<int> boundary;       ///< rank boundary
  IntArray2D       coord;          ///< chunk ID to coordinate map
  IntArray3D       chunkid;        ///< coordinate to chunk ID map

public:
  /// @brief constructor
  ChunkMap(int Cz, int Cy, int Cx);

  /// @brief constructor
  ChunkMap(const int dims[3]);

  /// @brief return true if the chunkamp is valid
  virtual bool validate();

  /// @brief return if the chunk is active
  virtual bool is_chunk_active(int id);

  /// @brief serialize to json object
  virtual json to_json();

  /// @brief deserialize from json object
  virtual void from_json(json& obj);

  /// @brief set periodicity for each direction
  virtual void set_periodicity(int pz, int py, int px);

  /// @brief return neighbor coordinate for a specific direction
  virtual int get_neighbor_coord(int coord, int delta, int dir);

  /// @brief get rank for chunk
  virtual int get_rank(int id);

  /// @brief set process rank boundary
  virtual void set_rank_boundary(std::vector<int>& boundary);

  /// @brief get process rank boundary
  virtual std::vector<int> get_rank_boundary();

  /// @brief get coordinate of chunk
  virtual std::tuple<int, int, int> get_coordinate(int id);

  /// @brief get chunk ID for coordinate
  virtual int get_chunkid(int cz, int cy, int cx);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
