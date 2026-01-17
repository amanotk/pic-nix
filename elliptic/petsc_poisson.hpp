#ifndef _PETSC_POISSON_HPP_
#define _PETSC_POISSON_HPP_

#include "elliptic.hpp"
#include "petsc_interface.hpp"

namespace elliptic
{
using namespace nix::typedefs;

class PetscPoisson : public PetscInterface
{
public:
  PetscPoisson(Dims3D dims, float64 delh);

  virtual int solve() override;
  virtual int solve(ChunkAccessor& accessor) override;

  float64 get_residual_norm();

protected:
  float64 delx;
  float64 dely;
  float64 delz;

  void set_nullspace();
};

/// @brief 1D Poisson solver with periodic boundary conditions
class PetscPoisson1D : public PetscPoisson
{
public:
  PetscPoisson1D(Dims3D dims, float64 delh);

protected:
  virtual int set_matrix() override;
};

/// @brief 2D Poisson solver with periodic boundary conditions
class PetscPoisson2D : public PetscPoisson
{
public:
  PetscPoisson2D(Dims3D dims, float64 delh);

protected:
  virtual int set_matrix() override;
};

/// @brief 3D Poisson solver with periodic boundary conditions
class PetscPoisson3D : public PetscPoisson
{
public:
  PetscPoisson3D(Dims3D dims, float64 delh);

protected:
  virtual int set_matrix() override;
};

} // namespace elliptic

#endif //_PETSC_POISSON_HPP_
