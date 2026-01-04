#ifndef _PETSC_SOLVER_HPP_
#define _PETSC_SOLVER_HPP_

#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

namespace elliptic
{

using namespace nix::typedefs;

class PetscSolver
{
public:
  static void initialize_dmda(DM& dm_obj, std::vector<int> dims)
  {
    assert(dims.size() == 3);

    // check dimension and initialize DM object
    bool is_3d   = (dims[0] >= 2) && (dims[1] >= 2) && (dims[2] >= 2);
    bool is_2d   = (dims[0] == 1) && (dims[1] >= 2) && (dims[2] >= 2);
    bool is_1d   = (dims[0] == 1) && (dims[1] == 1) && (dims[2] >= 2);
    bool invalid = (is_3d == false) && (is_2d == false) && (is_1d == false);

    if (invalid == true) {
      // invalid dimension
      ERROR << "Invalid dimension is specified for PetscSolver::initialize_dmda()";
    } else if (is_1d == true) {
      DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, dims[2], 1, 1, nullptr, &dm_obj);
    } else if (is_2d == true) {
      DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, dims[2],
                   dims[1], PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm_obj);

    } else if (is_3d == true) {
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                   DMDA_STENCIL_BOX, dims[2], dims[1], dims[0], PETSC_DECIDE, PETSC_DECIDE,
                   PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
    }

    DMSetUp(dm_obj);
  }
};

} // namespace elliptic

#endif
