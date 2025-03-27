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
  ChunkMap(int Cz, int Cy, int Cx) : periodicity{1, 1, 1}
  {
    size    = Cz * Cy * Cx;
    dims[0] = Cz;
    dims[1] = Cy;
    dims[2] = Cx;

    // memory allocation
    {
      std::vector<size_t> dims1 = {static_cast<size_t>(size)};
      std::vector<size_t> dims2 = {static_cast<size_t>(size), 3};
      std::vector<size_t> dims3 = {static_cast<size_t>(Cz), static_cast<size_t>(Cy),
                                   static_cast<size_t>(Cx)};

      coord.resize(dims2);
      chunkid.resize(dims3);

      coord.fill(0);
      chunkid.fill(0);
    }

    // build mapping
    sfc::get_map3d(Cz, Cy, Cx, chunkid, coord);
  }

  ///
  /// @brief constructor
  /// @param dims number of chunk in each direction
  ///
  ChunkMap(const int dims[3]) : ChunkMap(dims[0], dims[1], dims[2])
  {
  }

  ///
  /// @brief check the validity of map
  /// @return true if it is valid map, false otherwise
  ///
  virtual bool validate()
  {
    return sfc::check_index(chunkid) & sfc::check_locality3d(coord);
  }

  ///
  /// @brief get map information as json object
  /// @return obj json object
  ///
  virtual json to_json()
  {
    json obj;

    // meta data
    obj["size"]        = size;
    obj["ndim"]        = 3;
    obj["shape"]       = {dims[0], dims[1], dims[2]};
    obj["periodicity"] = {periodicity[0], periodicity[1], periodicity[2]};

    // map
    obj["chunkid"]  = chunkid;
    obj["coord"]    = coord;
    obj["boundary"] = boundary;

    return obj;
  }

  ///
  /// @brief restore map information from json object
  /// @param obj json object
  ///
  virtual void from_json(json& obj)
  {
    if (obj["ndim"].get<int>() != 3) {
      ERROR << tfm::format("Invalid input to ChunkMap::load_json");
    }

    // meta data
    size           = obj["size"].get<int>();
    dims[0]        = obj["shape"][0].get<int>();
    dims[1]        = obj["shape"][1].get<int>();
    dims[2]        = obj["shape"][2].get<int>();
    periodicity[0] = obj["periodicity"][0].get<int>();
    periodicity[1] = obj["periodicity"][1].get<int>();
    periodicity[2] = obj["periodicity"][2].get<int>();

    // map
    chunkid  = obj["chunkid"];
    coord    = obj["coord"];
    boundary = obj["boundary"].get<std::vector<int>>();
  }

  ///
  /// @brief set periodicity in each direction
  /// @param pz periodicity in z direction
  /// @param py periodicity in y direction
  /// @param px periodicity in x direction
  ///
  virtual void set_periodicity(int pz, int py, int px)
  {
    periodicity[0] = pz;
    periodicity[1] = py;
    periodicity[2] = px;
  }

  ///
  /// @brief return neighbor coordinate for a specific direction `dir`
  /// @param coord index of coordinate
  /// @param delta difference of index of coordinate from `coord`
  /// @param dir direction of coordinate
  /// @return `coord + delta` if not at boundary, otherwise boundary condition dependent
  ///
  virtual int get_neighbor_coord(int coord, int delta, int dir)
  {
    int cdir = coord + delta;

    if (periodicity[dir] == 1) {
      cdir = cdir >= 0 ? cdir : dims[dir] - 1;
      cdir = cdir < dims[dir] ? cdir : 0;
    } else {
      cdir = (cdir >= 0 && cdir < dims[dir]) ? cdir : -1;
    }

    return cdir;
  }

  ///
  /// @brief get process rank associated with chunk ID
  /// @param id chunk ID
  /// @return rank
  ///
  virtual int get_rank(int id)
  {
    if (id >= 0 && id < size) {
      auto it = std::upper_bound(boundary.begin(), boundary.end(), id);
      return std::distance(boundary.begin(), it) - 1;
    } else {
      return MPI_PROC_NULL;
    }
  }

  ///
  /// @brief set process rank boundary
  /// @param boundary array of rank boundary to set
  ///
  virtual void set_rank_boundary(std::vector<int>& boundary)
  {
    this->boundary = boundary;
  }

  ///
  /// @brief get process rank boundary
  /// @return array of rank boundary
  ///
  virtual std::vector<int> get_rank_boundary()
  {
    return boundary;
  }
  ///
  /// @brief get coordinate of chunk for 3D map
  /// @param id chunk ID
  /// @param cz z coordinate of chunk will be stored
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  virtual void get_coordinate(int id, int& cz, int& cy, int& cx)
  {
    if (id >= 0 && id < size) {
      cx = coord(id, 0);
      cy = coord(id, 1);
      cz = coord(id, 2);
    } else {
      cz = -1;
      cy = -1;
      cx = -1;
    }
  }

  ///
  /// @brief get chunk ID for 3D map
  /// @param cz z coordinate of chunk
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  virtual int get_chunkid(int cz, int cy, int cx)
  {
    if ((cz >= 0 && cz < dims[0]) && (cy >= 0 && cy < dims[1]) && (cx >= 0 && cx < dims[2])) {
      return chunkid(cz, cy, cx);
    } else {
      return -1;
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
