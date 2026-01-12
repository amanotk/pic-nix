// -*- C++-*-
#ifndef _ELLIPTIC_HPP_
#define _ELLIPTIC_HPP_

#include <memory>

#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "petsc_solver.hpp"
#include "petsc_scatter.hpp"

namespace elliptic
{

using namespace nix::typedefs;

template <typename T_chunkvec>
class Solver
{
public:
  using T_chunk_ptr = typename T_chunkvec::value_type;

protected:
  // PETSc objects
  DM  dm_obj;
  Vec vec_src_l;
  Vec vec_src_g;
  Vec vec_sol_l;
  Vec vec_sol_g;

  std::unique_ptr<PetscScatter> scatter;

  std::vector<int>     dims;
  std::vector<float64> src_local;
  std::vector<float64> sol_local;

public:
  Solver(int dims[3])
      : dm_obj(nullptr), vec_src_l(nullptr), vec_src_g(nullptr), vec_sol_l(nullptr),
        vec_sol_g(nullptr)
  {
    this->dims.resize(3);
    this->dims[0] = dims[0];
    this->dims[1] = dims[1];
    this->dims[2] = dims[2];

    PetscSolver::initialize_dmda(dm_obj, this->dims);
    scatter = std::make_unique<PetscScatter>(&dm_obj);

    // create global vectors
    DMCreateGlobalVector(dm_obj, &vec_src_g);
    DMCreateGlobalVector(dm_obj, &vec_sol_g);
  }

  static void initialize()
  {
    PetscInitialize(NULL, NULL, NULL, NULL);
  }

  static void finalize()
  {
    PetscFinalize();
  }

  int scatter_forward_begin()
  {
    scatter->scatter_forward_begin(vec_src_l, vec_src_g);

    return 0;
  }

  int scatter_forward_end()
  {
    scatter->scatter_forward_end(vec_src_l, vec_src_g);

    return 0;
  }

  int scatter_reverse_begin()
  {
    scatter->scatter_reverse_begin(vec_sol_l, vec_sol_g);

    return 0;
  }

  int scatter_reverse_end()
  {
    scatter->scatter_reverse_end(vec_sol_l, vec_sol_g);

    return 0;
  }

  int update_mapping(T_chunkvec& chunkvec)
  {
    assert(chunkvec.size() > 0);

    auto                  chunk_dims = chunkvec[0]->get_dims();
    int                   chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
    int                   num_grids  = chunk_size * chunkvec.size();
    std::vector<PetscInt> index(num_grids);

    // global index for the local data
    PetscScatter::calc_global_index(chunkvec, dims, index);

    // local vectors
    src_local.resize(num_grids);
    sol_local.resize(num_grids);
    scatter->setup_vector_local(src_local, vec_src_l);
    scatter->setup_vector_local(sol_local, vec_sol_l);

    // scatter object
    scatter->setup_indexset_local(index.size());
    scatter->setup_indexset_global(index);
    scatter->setup_scatter(vec_src_l, vec_src_g);

    return 0;
  }

  int copy_chunk_to_src(T_chunkvec& chunkvec)
  {
    assert(chunkvec.size() > 0);

    auto             chunk_dims = chunkvec[0]->get_dims();
    int              chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
    std::vector<int> lstride    = {chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};

    for (int i = 0; i < chunkvec.size(); ++i) {
      auto data = chunkvec[i]->get_internal_data();

      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            int jz = iz - data.Lbz;
            int jy = iy - data.Lby;
            int jx = ix - data.Lbx;
            int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size;

            src_local[jj] = data.uj(iz, iy, ix, 0);
          }
        }
      }
    }

    return 0;
  }

  int copy_sol_to_chunk(T_chunkvec& chunkvec)
  {
    assert(chunkvec.size() > 0);

    auto             chunk_dims = chunkvec[0]->get_dims();
    int              chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
    std::vector<int> lstride    = {chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};

    for (int i = 0; i < chunkvec.size(); ++i) {
      auto data = chunkvec[i]->get_internal_data();

      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            int jz = iz - data.Lbz;
            int jy = iy - data.Lby;
            int jx = ix - data.Lbx;
            int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size;

            data.uj(iz, iy, ix, 0) = sol_local[jj];
          }
        }
      }
    }

    return 0;
  }
};

} // namespace elliptic

#endif
