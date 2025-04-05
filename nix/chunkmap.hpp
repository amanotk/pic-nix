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
/// The chunk ID is defined with row-major ordering of chunks in cartesian
/// coordinate. Mapping between chunk ID and cartesian coordinate may be
/// calculated via get_chunk() and get_coordinate() methods.
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
  ///
  /// @brief constructor for 3D map
  /// @param Cz number of chunk in z direction
  /// @param Cy number of chunk in y direction
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(int Cz, int Cy, int Cx);

  ///
  /// @brief constructor
  /// @param dims number of chunk in each direction
  ///
  ChunkMap(const int dims[3]);

  ///
  /// @brief check the validity of map
  /// @return true if it is valid map, false otherwise
  ///
  virtual bool validate();

  ///
  /// @brief return if the chunk is active
  ///
  virtual bool is_chunk_active(int id);

  ///
  /// @brief get map information as json object
  /// @return obj json object
  ///
  virtual json to_json();

  ///
  /// @brief restore map information from json object
  /// @param obj json object
  ///
  virtual void from_json(json& obj);

  ///
  /// @brief set periodicity in each direction
  /// @param pz periodicity in z direction
  /// @param py periodicity in y direction
  /// @param px periodicity in x direction
  ///
  virtual void set_periodicity(int pz, int py, int px);

  ///
  /// @brief return neighbor coordinate for a specific direction `dir`
  /// @param coord index of coordinate
  /// @param delta difference of index of coordinate from `coord`
  /// @param dir direction of coordinate
  /// @return `coord + delta` if not at boundary, otherwise boundary condition dependent
  ///
  virtual int get_neighbor_coord(int coord, int delta, int dir);

  ///
  /// @brief get process rank associated with chunk ID
  /// @param id chunk ID
  /// @return rank
  ///
  virtual int get_rank(int id);

  ///
  /// @brief set process rank boundary
  /// @param boundary array of rank boundary to set
  ///
  virtual void set_rank_boundary(std::vector<int>& boundary);

  ///
  /// @brief get process rank boundary
  /// @return array of rank boundary
  ///
  virtual std::vector<int> get_rank_boundary();

  ///
  /// @brief get coordinate of chunk for 3D map
  /// @param id chunk ID
  /// @param cz z coordinate of chunk will be stored
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  virtual std::tuple<int, int, int> get_coordinate(int id);

  ///
  /// @brief get chunk ID for 3D map
  /// @param cz z coordinate of chunk
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  virtual int get_chunkid(int cz, int cy, int cx);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
