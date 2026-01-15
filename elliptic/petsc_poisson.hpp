#ifndef _PETSC_POISSON_HPP_
#define _PETSC_POISSON_HPP_

#include "elliptic.hpp"
#include "petsc_interface.hpp"

namespace elliptic
{
using namespace nix::typedefs;
using nix::Dims3D;

class PetscPoisson : public PetscInterface
{
public:
  PetscPoisson(Dims3D dims, float64 delh) : PetscInterface(dims), delx(delh), dely(delh), delz(delh)
  {
  }

  virtual int solve(ChunkAccessor& accessor) override
  {
    PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
    if (ierr != PETSC_SUCCESS) {
      ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
    }
    return ierr;
  }

  float64 get_residual_norm()
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

protected:
  float64 delx;
  float64 dely;
  float64 delz;
};

/// @brief 1D Poisson solver with periodic boundary conditions
class PetscPoisson1D : public PetscPoisson
{
public:
  PetscPoisson1D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
  {
    setup();
  }

protected:
  virtual int set_matrix() override
  {
    const float64 dx2_inv = 1.0 / (delx * delx);
    const float64 diag    = +2.0 * dx2_inv;
    const float64 ofdx    = -1.0 * dx2_inv;

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    if (matrix != nullptr) {
      MatDestroy(&matrix);
    }
    DMCreateMatrix(dm_obj, &matrix);

    for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
      MatStencil row;
      MatStencil col[3];
      float64    vals[3];
      int        ncols = 0;

      row.i = ix;
      row.j = 0;
      row.k = 0;
      row.c = 0;

      col[ncols]    = row;
      vals[ncols++] = diag;

      col[ncols].i  = ix - 1;
      col[ncols].j  = 0;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdx;

      col[ncols].i  = ix + 1;
      col[ncols].j  = 0;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdx;

      MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, INSERT_VALUES);
    }

    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);

    // set null space
    {
      MatNullSpace ns;
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &ns);
      MatSetNullSpace(matrix, ns);
      MatNullSpaceDestroy(&ns);
    }

    return 0;
  }
};

/// @brief 2D Poisson solver with periodic boundary conditions
class PetscPoisson2D : public PetscPoisson
{
public:
  PetscPoisson2D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
  {
    setup();
  }

protected:
  virtual int set_matrix() override
  {
    const float64 dx2_inv = 1.0 / (delx * delx);
    const float64 dy2_inv = 1.0 / (dely * dely);
    const float64 diag    = +2.0 * dx2_inv + 2.0 * dy2_inv;
    const float64 ofdx    = -1.0 * dx2_inv;
    const float64 ofdy    = -1.0 * dy2_inv;

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    if (matrix != nullptr) {
      MatDestroy(&matrix);
    }
    DMCreateMatrix(dm_obj, &matrix);

    for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
      for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
        MatStencil row;
        MatStencil col[5];
        float64    vals[5];
        int        ncols = 0;

        row.i = ix;
        row.j = iy;
        row.k = 0;
        row.c = 0;

        col[ncols]    = row;
        vals[ncols++] = diag;

        col[ncols].i  = ix - 1;
        col[ncols].j  = iy;
        col[ncols].k  = 0;
        col[ncols].c  = 0;
        vals[ncols++] = ofdx;

        col[ncols].i  = ix + 1;
        col[ncols].j  = iy;
        col[ncols].k  = 0;
        col[ncols].c  = 0;
        vals[ncols++] = ofdx;

        col[ncols].i  = ix;
        col[ncols].j  = iy - 1;
        col[ncols].k  = 0;
        col[ncols].c  = 0;
        vals[ncols++] = ofdy;

        col[ncols].i  = ix;
        col[ncols].j  = iy + 1;
        col[ncols].k  = 0;
        col[ncols].c  = 0;
        vals[ncols++] = ofdy;

        MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, INSERT_VALUES);
      }
    }

    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);

    // set null space
    {
      MatNullSpace ns;
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &ns);
      MatSetNullSpace(matrix, ns);
      MatNullSpaceDestroy(&ns);
    }

    return 0;
  }
};

/// @brief 3D Poisson solver with periodic boundary conditions
class PetscPoisson3D : public PetscPoisson
{
public:
  PetscPoisson3D(Dims3D dims, float64 delh) : PetscPoisson(dims, delh)
  {
    setup();
  }

protected:
  virtual int set_matrix() override
  {
    const float64 dx2_inv = 1.0 / (delx * delx);
    const float64 dy2_inv = 1.0 / (dely * dely);
    const float64 dz2_inv = 1.0 / (delz * delz);
    const float64 diag    = +2.0 * dx2_inv + 2.0 * dy2_inv + 2.0 * dz2_inv;
    const float64 ofdx    = -1.0 * dx2_inv;
    const float64 ofdy    = -1.0 * dy2_inv;
    const float64 ofdz    = -1.0 * dz2_inv;

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    // create matrix
    if (matrix != nullptr) {
      MatDestroy(&matrix);
    }
    DMCreateMatrix(dm_obj, &matrix);

    for (int iz = info.zs; iz < info.zs + info.zm; ++iz) {
      for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
        for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
          MatStencil row;
          MatStencil col[7];
          float64    vals[7];
          int        ncols = 0;

          row.i = ix;
          row.j = iy;
          row.k = iz;
          row.c = 0;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = diag;
          ncols++;

          col[ncols].i = ix - 1;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdx;
          ncols++;

          col[ncols].i = ix + 1;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdx;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy - 1;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdy;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy + 1;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdy;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz - 1;
          col[ncols].c = 0;
          vals[ncols]  = ofdz;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz + 1;
          col[ncols].c = 0;
          vals[ncols]  = ofdz;
          ncols++;

          MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, INSERT_VALUES);
        }
      }
    }

    // assemble matrix
    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);

    // set null space
    {
      MatNullSpace ns;
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &ns);
      MatSetNullSpace(matrix, ns);
      MatNullSpaceDestroy(&ns);
    }

    return 0;
  }
};

} // namespace elliptic

#endif //_PETSC_POISSON_HPP_