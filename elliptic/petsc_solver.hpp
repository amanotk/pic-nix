#ifndef _PETSC_SOLVER_HPP_
#define _PETSC_SOLVER_HPP_

#include "nix.hpp"
#include <iomanip>
#include <limits>
#include <memory> // add
#include <optional>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "elliptic.hpp"
#include "petsc_scatter.hpp"

namespace elliptic
{

using namespace nix::typedefs;

class PetscInterface : public SolverInterface
{
public:
  using Option     = std::pair<std::string, std::string>;
  using OptionVec  = std::vector<Option>;
  using PtrScatter = std::unique_ptr<PetscScatter>;

  explicit PetscInterface(Dims3D dims);
  virtual ~PetscInterface();

protected:
  Dims3D               dims;
  std::vector<float64> src_local;
  std::vector<float64> sol_local;
  DM                   dm_obj;
  KSP                  ksp_obj;
  Mat                  matrix;
  Vec                  vector_src_l;
  Vec                  vector_src_g;
  Vec                  vector_sol_l;
  Vec                  vector_sol_g;
  OptionVec            option;
  PtrScatter           scatter;

  static void        initialize();
  static void        finalize();
  static std::string bool_to_string(bool x);
  static std::string int_to_string(int x);
  static std::string float_to_string(double x);
  static int         apply_petsc_option(const OptionVec& opts);
  static OptionVec   make_petsc_option(const nlohmann::json& config);
  static OptionVec   make_petsc_option(const toml::value& config);

  int scatter_forward_begin();
  int scatter_forward_end();
  int scatter_reverse_begin();
  int scatter_reverse_end();

  virtual void create_dm(Dims3D dims);
  virtual void create_dm1d(Dims3D dims);
  virtual void create_dm2d(Dims3D dims);
  virtual void create_dm3d(Dims3D dims);
  virtual void set_matrix(float64 hx, float64 hy, float64 hz) = 0;
  virtual int  update_mapping(ChunkAccessor& accessor);
  virtual int  copy_chunk_to_src(ChunkAccessor& accessor);
  virtual int  copy_sol_to_chunk(ChunkAccessor& accessor);
};

class Poisson3D : public PetscInterface
{
public:
  Poisson3D(Dims3D dims) : PetscInterface(dims)
  {
    set_matrix(1.0, 1.0, 1.0);
  }

protected:
  void set_matrix(float64 hx, float64 hy, float64 hz) override
  {
    const float64 hx2_inv = 1.0 / (hx * hx);
    const float64 hy2_inv = 1.0 / (hy * hy);
    const float64 hz2_inv = 1.0 / (hz * hz);
    const float64 diag    = 2.0 * hx2_inv + 2.0 * hy2_inv + 2.0 * hz2_inv;
    const float64 ofdx    = -1.0 * hx2_inv;
    const float64 ofdy    = -1.0 * hy2_inv;
    const float64 ofdz    = -1.0 * hz2_inv;

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
  }
};

} // namespace elliptic

#endif
