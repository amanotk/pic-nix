#include "petsc_matrix_helpers.hpp"

#include "petsc_poisson.hpp"

#include <petscdmda.h>

namespace elliptic
{

using namespace nix::typedefs;

PetscPoisson::PetscPoisson(Dims3D dims, float64 delh)
    : PetscInterface(dims), delx(delh), dely(delh), delz(delh)
{
}

int PetscPoisson::solve(ChunkAccessor& accessor)
{
  PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
  if (ierr != 0) {
    ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
  }
  return ierr;
}

void PetscPoisson::set_nullspace()
{
  MatNullSpace ns;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &ns);
  MatSetNullSpace(matrix, ns);
  MatNullSpaceDestroy(&ns);
}

PetscPoisson1D::PetscPoisson1D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
{
  setup();
}

int PetscPoisson1D::set_matrix()
{
  const float64 dx2_inv = 1.0 / (delx * delx);
  const float64 diag    = +2.0 * dx2_inv;
  const float64 ofdx    = -1.0 * dx2_inv;

  build_poisson_matrix_1d(matrix, dm_obj, diag, ofdx);
  return 0;
}

PetscPoisson2D::PetscPoisson2D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
{
  setup();
}

int PetscPoisson2D::set_matrix()
{
  const float64 dx2_inv = 1.0 / (delx * delx);
  const float64 dy2_inv = 1.0 / (dely * dely);
  const float64 diag    = +2.0 * dx2_inv + 2.0 * dy2_inv;
  const float64 ofdx    = -1.0 * dx2_inv;
  const float64 ofdy    = -1.0 * dy2_inv;

  build_poisson_matrix_2d(matrix, dm_obj, diag, ofdx, ofdy);
  return 0;
}

PetscPoisson3D::PetscPoisson3D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
{
  setup();
}

int PetscPoisson3D::set_matrix()
{
  const float64 dx2_inv = 1.0 / (delx * delx);
  const float64 dy2_inv = 1.0 / (dely * dely);
  const float64 dz2_inv = 1.0 / (delz * delz);
  const float64 diag    = +2.0 * dx2_inv + 2.0 * dy2_inv + 2.0 * dz2_inv;
  const float64 ofdx    = -1.0 * dx2_inv;
  const float64 ofdy    = -1.0 * dy2_inv;
  const float64 ofdz    = -1.0 * dz2_inv;

  build_poisson_matrix_3d(matrix, dm_obj, diag, ofdx, ofdy, ofdz);
  return 0;
}

} // namespace elliptic
