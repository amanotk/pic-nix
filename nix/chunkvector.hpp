// -*- C++ -*-
#ifndef _CHUNKVEC_HPP_
#define _CHUNKVEC_HPP_

#include "nix.hpp"

#include "chunkmap.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief ChunkVector class
///
/// A convenient wrapper of chunk array.
///
template <typename PtrChunk>
class ChunkVector : private std::vector<PtrChunk>
{
public:
  using vector_type = std::vector<PtrChunk>;
  using vector_type::push_back;
  using vector_type::emplace_back;
  using vector_type::begin;
  using vector_type::end;
  using vector_type::rbegin;
  using vector_type::rend;
  using vector_type::front;
  using vector_type::back;
  using vector_type::operator[];
  using vector_type::size;
  using vector_type::capacity;
  using vector_type::resize;
  using vector_type::reserve;
  using vector_type::shrink_to_fit;

  void remove_nullptr()
  {
    this->erase(
        std::remove_if(this->begin(), this->end(), [](const auto& x) { return x == nullptr; }),
        this->end());
  }

  void sort_and_shrink()
  {
    // clean up
    this->remove_nullptr();

    // sort by id
    std::sort(this->begin(), this->end(),
              [](const auto& x, const auto& y) { return x->get_id() < y->get_id(); });

    this->shrink_to_fit();
  }

  void set_neighbors(std::unique_ptr<ChunkMap>& chunkmap)
  {
    for (int i = 0; i < this->size(); i++) {
      int ix = 0;
      int iy = 0;
      int iz = 0;
      int id = (*this)[i]->get_id();
      chunkmap->get_coordinate(id, iz, iy, ix);

      for (int dirz = -1; dirz <= +1; dirz++) {
        for (int diry = -1; diry <= +1; diry++) {
          for (int dirx = -1; dirx <= +1; dirx++) {
            // neighbor coordinate
            int cz = chunkmap->get_neighbor_coord(iz, dirz, 0);
            int cy = chunkmap->get_neighbor_coord(iy, diry, 1);
            int cx = chunkmap->get_neighbor_coord(ix, dirx, 2);

            // set neighbor id
            int nbid = chunkmap->get_chunkid(cz, cy, cx);
            (*this)[i]->set_nb_id(dirz, diry, dirx, nbid);

            // set neighbor rank
            int nbrank = chunkmap->get_rank(nbid);
            (*this)[i]->set_nb_rank(dirz, diry, dirx, nbrank);
          }
        }
      }
    }
  }

  bool validate(std::unique_ptr<ChunkMap>& chunkmap)
  {
    bool status = true;

    // check local number of chunks
    if (size() > MAX_CHUNK_PER_RANK) {
      ERROR << tfm::format("Number of chunk per rank should not exceed %8d", MAX_CHUNK_PER_RANK);
      status = status & false;
    } else {
      status = status & true;
    }

    // check neighbor rank and ID
    for (int i = 0; i < size(); i++) {
      int ix = 0;
      int iy = 0;
      int iz = 0;
      int id = (*this)[i]->get_id();
      chunkmap->get_coordinate(id, iz, iy, ix);

      // check neighbor ID and rank
      for (int dirz = -1; dirz <= +1; dirz++) {
        for (int diry = -1; diry <= +1; diry++) {
          for (int dirx = -1; dirx <= +1; dirx++) {
            // neighbor coordinate
            int cz = chunkmap->get_neighbor_coord(iz, dirz, 0);
            int cy = chunkmap->get_neighbor_coord(iy, diry, 1);
            int cx = chunkmap->get_neighbor_coord(ix, dirx, 2);

            int nbid   = (*this)[i]->get_nb_id(dirz, diry, dirx);
            int nbrank = (*this)[i]->get_nb_rank(dirz, diry, dirx);
            int id     = chunkmap->get_chunkid(cz, cy, cx);
            int rank   = chunkmap->get_rank(id);

            status = status & (id == nbid);
            status = status & (rank == nbrank);
          }
        }
      }
    }

    return status;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
