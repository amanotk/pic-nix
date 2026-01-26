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
  return solve();
}

int PetscPoisson::solve()
{
  PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
  if (ierr != 0) {
    ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
  }
  return ierr;
}

float64 PetscPoisson::get_residual_norm()
{
  Vec       vector_res_g;
  PetscReal res_norm;
  PetscReal src_norm;

  VecDuplicate(vector_src_g, &vector_res_g);
  MatMult(matrix, vector_sol_g, vector_res_g);
  VecAYPX(vector_res_g, -1.0, vector_src_g);
  VecNorm(vector_res_g, NORM_2, &res_norm);
  VecNorm(vector_src_g, NORM_2, &src_norm);
  VecDestroy(&vector_res_g);

  return static_cast<float64>(res_norm / (src_norm + 1.0e-32));
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

  if (matrix != nullptr) {
    MatDestroy(&matrix);
  }
  DMCreateMatrix(dm_obj, &matrix);

  build_poisson_matrix_1d(matrix, dm_obj, diag, ofdx);

  set_nullspace();

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

  if (matrix != nullptr) {
    MatDestroy(&matrix);
  }
  DMCreateMatrix(dm_obj, &matrix);

  build_poisson_matrix_2d(matrix, dm_obj, diag, ofdx, ofdy);

  set_nullspace();

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

  if (matrix != nullptr) {
    MatDestroy(&matrix);
  }
  DMCreateMatrix(dm_obj, &matrix);

  build_poisson_matrix_3d(matrix, dm_obj, diag, ofdx, ofdy, ofdz);

  set_nullspace();

  return 0;
}

} // namespace elliptic
