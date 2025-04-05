// -*- C++ -*-
#include "chunkmap.hpp"

NIX_NAMESPACE_BEGIN

ChunkMap::ChunkMap(int Cz, int Cy, int Cx) : periodicity{1, 1, 1}
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

ChunkMap::ChunkMap(const int dims[3]) : ChunkMap(dims[0], dims[1], dims[2])
{
}

bool ChunkMap::validate()
{
  return sfc::check_index(chunkid) & sfc::check_locality3d(coord);
}

bool ChunkMap::is_chunk_active(int id)
{
  return true;
}

json ChunkMap::to_json()
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

void ChunkMap::from_json(json& obj)
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

void ChunkMap::set_periodicity(int pz, int py, int px)
{
  periodicity[0] = pz;
  periodicity[1] = py;
  periodicity[2] = px;
}

int ChunkMap::get_neighbor_coord(int coord, int delta, int dir)
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

int ChunkMap::get_rank(int id)
{
  if (id >= 0 && id < size) {
    auto it = std::upper_bound(boundary.begin(), boundary.end(), id);
    return std::distance(boundary.begin(), it) - 1;
  } else {
    return MPI_PROC_NULL;
  }
}

void ChunkMap::set_rank_boundary(std::vector<int>& boundary)
{
  this->boundary = boundary;
}

std::vector<int> ChunkMap::get_rank_boundary()
{
  return boundary;
}

std::tuple<int, int, int> ChunkMap::get_coordinate(int id)
{
  int cz, cy, cx;

  if (id >= 0 && id < size) {
    cx = coord(id, 0);
    cy = coord(id, 1);
    cz = coord(id, 2);
  } else {
    cz = -1;
    cy = -1;
    cx = -1;
  }

  return std::make_tuple(cz, cy, cx);
}

int ChunkMap::get_chunkid(int cz, int cy, int cx)
{
  if ((cz >= 0 && cz < dims[0]) && (cy >= 0 && cy < dims[1]) && (cx >= 0 && cx < dims[2])) {
    return chunkid(cz, cy, cx);
  } else {
    return -1;
  }
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
