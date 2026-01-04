// -*- C++-*-
#ifndef _PETSC_UTILS_HPP_
#define _PETSC_UTILS_HPP_

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

// ensure compatibility of PetscInt and int
static_assert(sizeof(PetscInt) == sizeof(int), "PetscInt and int must have the same size");

namespace elliptic
{

using namespace nix::typedefs;

class PetscUtils
{
protected:
  DM*        dm_ptr; // reference to DM object
  VecScatter sc_obj;
  IS         is_obj_l;
  IS         is_obj_g;

public:
  PetscUtils(DM* dm) : dm_ptr(dm), sc_obj(nullptr), is_obj_l(nullptr), is_obj_g(nullptr)
  {
  }

  virtual ~PetscUtils()
  {
    if (sc_obj != nullptr) {
      VecScatterDestroy(&sc_obj);
    }
    if (is_obj_l != nullptr) {
      ISDestroy(&is_obj_l);
    }
    if (is_obj_g != nullptr) {
      ISDestroy(&is_obj_g);
    }
  }

  int scatter_forward_begin(Vec& vec_local, Vec& vec_global)
  {
    VecScatterBegin(sc_obj, vec_local, vec_global, INSERT_VALUES, SCATTER_FORWARD);

    return 0;
  }

  int scatter_forward_end(Vec& vec_local, Vec& vec_global)
  {
    VecScatterEnd(sc_obj, vec_local, vec_global, INSERT_VALUES, SCATTER_FORWARD);

    return 0;
  }

  int scatter_reverse_begin(Vec& vec_local, Vec& vec_global)
  {
    VecScatterBegin(sc_obj, vec_local, vec_global, INSERT_VALUES, SCATTER_REVERSE);

    return 0;
  }

  int scatter_reverse_end(Vec& vec_local, Vec& vec_global)
  {
    VecScatterEnd(sc_obj, vec_local, vec_global, INSERT_VALUES, SCATTER_REVERSE);

    return 0;
  }

  int setup_vector_local(std::vector<float64>& buffer, Vec& vec)
  {
    if (vec != nullptr) {
      VecDestroy(&vec);
    }

    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, buffer.size(), buffer.data(), &vec);

    return 0;
  }

  int setup_indexset_local(std::vector<PetscInt>& index)
  {
    if (is_obj_l != nullptr) {
      ISDestroy(&is_obj_l);
    }

    ISCreateStride(PETSC_COMM_SELF, index.size(), 0, 1, &is_obj_l);

    return 0;
  }

  int setup_indexset_global(std::vector<PetscInt>& index)
  {
    if (is_obj_g != nullptr) {
      ISDestroy(&is_obj_g);
    }

    AO        ao_obj;
    int       size = index.size();
    PetscInt* data = index.data();

    DMDAGetAO(*dm_ptr, &ao_obj); // DO NOT destroy it!
    AOApplicationToPetsc(ao_obj, size, data);
    ISCreateGeneral(PETSC_COMM_WORLD, size, data, PETSC_COPY_VALUES, &is_obj_g);

    return 0;
  }

  int setup_scatter(Vec& vec_local, Vec& vec_global)
  {
    if (sc_obj != nullptr) {
      VecScatterDestroy(&sc_obj);
    }

    VecScatterCreate(vec_local, is_obj_l, vec_global, is_obj_g, &sc_obj);

    return 0;
  }

  template <typename T_array>
  static int flatten_index(int iz, int iy, int ix, const T_array& dims)
  {
    const int stride_z = dims[1] * dims[2];
    const int stride_y = dims[2];
    const int stride_x = 1;

    return iz * stride_z + iy * stride_y + ix * stride_x;
  }

  template <typename T_chunkvec, typename T_array>
  static int calc_global_index(T_chunkvec& chunkvec, const T_array& dims, std::vector<int>& index)
  {
    assert(chunkvec.size() > 0);

    auto chunk_dims = chunkvec[0]->get_dims();
    int  chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];

    for (int i = 0; i < chunkvec.size(); ++i) {
      auto offset = chunkvec[i]->get_offset();

      for (int iz = 0; iz < chunk_dims[0]; ++iz) {
        for (int iy = 0; iy < chunk_dims[1]; ++iy) {
          for (int ix = 0; ix < chunk_dims[2]; ++ix) {
            int jz = iz + offset[0];
            int jy = iy + offset[1];
            int jx = ix + offset[2];
            int jj = flatten_index(iz, iy, ix, chunk_dims) + i * chunk_size;

            index[jj] = flatten_index(jz, jy, jx, dims);
          }
        }
      }
    }

    return 0;
  }
};

}; // namespace elliptic

#endif // _PETSC_UTILS_HPP_