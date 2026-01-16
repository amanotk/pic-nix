// -*- C++-*-
#ifndef _PETSC_SCATTER_HPP_
#define _PETSC_SCATTER_HPP_

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "chunk_accessor.hpp"
#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

// ensure compatibility
static_assert(sizeof(PetscInt) == sizeof(int), "PetscInt and int are incompatible");
static_assert(sizeof(PetscScalar) == sizeof(double), "PetscScalar and double are incompatible");

namespace elliptic
{

using namespace nix::typedefs;

class PetscScatter
{
protected:
  DM*        dm_ptr; // reference to DM object
  Dims3D     dims;
  VecScatter sc_obj;
  IS         is_obj_l;
  IS         is_obj_g;

private:
  int get_indexset(IS& is_obj, std::vector<int>& index);

public:
  PetscScatter(DM* dm, Dims3D dims);
  virtual ~PetscScatter();

  int scatter_forward_begin(Vec& src, Vec& dst);
  int scatter_forward_end(Vec& src, Vec& dst);
  int scatter_reverse_begin(Vec& src, Vec& dst);
  int scatter_reverse_end(Vec& src, Vec& dst);

  int setup_vector_local(std::vector<float64>& buffer, Vec& vec);
  int setup_indexset_local(int size);
  int setup_indexset_global(std::vector<int>& index);
  int setup_scatter(ChunkAccessor& accessor, std::vector<float64>& src, std::vector<float64>& sol,
                    Vec& vec_src, Vec& vec_sol, Vec& vec_global);
  int get_indexset_local(std::vector<int>& index);
  int get_indexset_global(std::vector<int>& index);
};

}; // namespace elliptic

#endif // _PETSC_SCATTER_HPP_