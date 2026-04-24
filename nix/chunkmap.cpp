// -*- C++ -*-
#include "chunkmap.hpp"

#include "sfc.hpp"

NIX_NAMESPACE_BEGIN

namespace
{
json chunkid_to_json(const std::vector<int>& chunkid, const int dims[3])
{
  json obj = json::array();

  for (int iz = 0; iz < dims[0]; iz++) {
    json plane = json::array();
    for (int iy = 0; iy < dims[1]; iy++) {
      json row = json::array();
      for (int ix = 0; ix < dims[2]; ix++) {
        row.push_back(chunkid[iz * dims[1] * dims[2] + iy * dims[2] + ix]);
      }
      plane.push_back(row);
    }
    obj.push_back(plane);
  }

  return obj;
}

json coord_to_json(const std::vector<int>& coord, int size)
{
  json obj = json::array();

  for (int id = 0; id < size; id++) {
    json row = json::array();
    row.push_back(coord[id * 3 + 0]);
    row.push_back(coord[id * 3 + 1]);
    row.push_back(coord[id * 3 + 2]);
    obj.push_back(row);
  }

  return obj;
}

std::vector<int> chunkid_from_json(const json& obj, const int dims[3])
{
  std::vector<int> chunkid(dims[0] * dims[1] * dims[2], 0);

  for (int iz = 0; iz < dims[0]; iz++) {
    for (int iy = 0; iy < dims[1]; iy++) {
      for (int ix = 0; ix < dims[2]; ix++) {
        chunkid[iz * dims[1] * dims[2] + iy * dims[2] + ix] = obj[iz][iy][ix].get<int>();
      }
    }
  }

  return chunkid;
}

std::vector<int> coord_from_json(const json& obj, int size)
{
  std::vector<int> coord(size * 3, 0);

  for (int id = 0; id < size; id++) {
    coord[id * 3 + 0] = obj[id][0].get<int>();
    coord[id * 3 + 1] = obj[id][1].get<int>();
    coord[id * 3 + 2] = obj[id][2].get<int>();
  }

  return coord;
}
} // namespace

ChunkMap::ChunkMap(int Cz, int Cy, int Cx) : periodicity{1, 1, 1}
{
  size    = Cz * Cy * Cx;
  dims[0] = Cz;
  dims[1] = Cy;
  dims[2] = Cx;

  coord.assign(size * 3, 0);
  chunkid.assign(size, 0);

  sfc::get_map3d(Cz, Cy, Cx, chunkid, coord);
}

ChunkMap::ChunkMap(const int dims[3]) : ChunkMap(dims[0], dims[1], dims[2])
{
}

bool ChunkMap::validate()
{
  return sfc::check_index(chunkid) & sfc::check_locality3d(coord, size);
}

bool ChunkMap::is_chunk_active(int id)
{
  return true;
}

json ChunkMap::to_json()
{
  json obj;

  obj["size"]        = size;
  obj["ndim"]        = 3;
  obj["shape"]       = {dims[0], dims[1], dims[2]};
  obj["periodicity"] = {periodicity[0], periodicity[1], periodicity[2]};
  obj["chunkid"]     = chunkid_to_json(chunkid, dims);
  obj["coord"]       = coord_to_json(coord, size);
  obj["boundary"]    = boundary;

  return obj;
}

void ChunkMap::from_json(json& obj)
{
  if (obj["ndim"].get<int>() != 3) {
    ERROR << tfm::format("Invalid input to ChunkMap::load_json");
  }

  size           = obj["size"].get<int>();
  dims[0]        = obj["shape"][0].get<int>();
  dims[1]        = obj["shape"][1].get<int>();
  dims[2]        = obj["shape"][2].get<int>();
  periodicity[0] = obj["periodicity"][0].get<int>();
  periodicity[1] = obj["periodicity"][1].get<int>();
  periodicity[2] = obj["periodicity"][2].get<int>();

  chunkid = chunkid_from_json(obj["chunkid"], dims);
  coord   = coord_from_json(obj["coord"], size);

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
    cx = coord[id * 3 + 0];
    cy = coord[id * 3 + 1];
    cz = coord[id * 3 + 2];
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
    return chunkid[cz * dims[1] * dims[2] + cy * dims[2] + cx];
  } else {
    return -1;
  }
}

NIX_NAMESPACE_END
