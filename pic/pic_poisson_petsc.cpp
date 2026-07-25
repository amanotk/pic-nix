// -*- C++ -*-
#include "pic_poisson_petsc.hpp"

#include "elliptic/petsc_interface.hpp"
#include "elliptic/petsc_matrix_helpers.hpp"

#include <petscdmda.h>
#include <petscmat.h>

class PicPoissonPetsc::Impl : public elliptic::PetscInterface
{
public:
  Impl(const nix::Dims3D& global_dims, float64 delh)
      : elliptic::PetscInterface(global_dims), delx(delh), dely(delh), delz(delh)
  {
    setup();
  }

  int solve(elliptic::ChunkAccessor& accessor) override
  {
    PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
    if (ierr != 0) {
      ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
    }
    return ierr;
  }

  int scatter_forward_begin()
  {
    return elliptic::PetscInterface::scatter_forward_begin();
  }

  int scatter_forward_end()
  {
    return elliptic::PetscInterface::scatter_forward_end();
  }

  int scatter_reverse_begin()
  {
    return elliptic::PetscInterface::scatter_reverse_begin();
  }

  int scatter_reverse_end()
  {
    return elliptic::PetscInterface::scatter_reverse_end();
  }

  Vec get_vector_src_g() const
  {
    return vector_src_g;
  }

  Vec get_vector_sol_g() const
  {
    return vector_sol_g;
  }

protected:
  int set_matrix() override
  {
    const bool    is_1d   = (dims[0] == 1) && (dims[1] == 1);
    const bool    is_2d   = (dims[0] == 1) && (dims[1] > 1);
    const bool    is_3d   = (dims[0] > 1) && (dims[1] > 1);
    const float64 dx2_inv = 1.0 / (delx * delx);
    const float64 dy2_inv = 1.0 / (dely * dely);
    const float64 dz2_inv = 1.0 / (delz * delz);
    const float64 ofdx    = -1.0 * dx2_inv;
    const float64 ofdy    = -1.0 * dy2_inv;
    const float64 ofdz    = -1.0 * dz2_inv;
    const float64 diag_1d = +2.0 * dx2_inv;
    const float64 diag_2d = +2.0 * dx2_inv + 2.0 * dy2_inv;
    const float64 diag_3d = +2.0 * dx2_inv + 2.0 * dy2_inv + 2.0 * dz2_inv;

    if (is_1d) {
      elliptic::build_poisson_matrix_1d(matrix, dm_obj, diag_1d, ofdx);
    } else if (is_2d) {
      elliptic::build_poisson_matrix_2d(matrix, dm_obj, diag_2d, ofdx, ofdy);
    } else if (is_3d) {
      elliptic::build_poisson_matrix_3d(matrix, dm_obj, diag_3d, ofdx, ofdy, ofdz);
    } else {
      ERROR << fmt::format("Invalid global dimensions for PicPoisson: {} {} {}", dims[0], dims[1],
                           dims[2]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    return 0;
  }

  void set_nullspace() override
  {
    MatNullSpace ns;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, &ns);
    MatSetNullSpace(matrix, ns);
    MatNullSpaceDestroy(&ns);
  }

private:
  float64 delx;
  float64 dely;
  float64 delz;
};

PicPoissonPetsc::PicPoissonPetsc(const nix::Dims3D& global_dims, float64 delh)
    : PicPoisson(global_dims, delh), impl(std::make_unique<Impl>(global_dims, delh))
{
}

PicPoissonPetsc::~PicPoissonPetsc() = default;

int PicPoissonPetsc::update_mapping(elliptic::ChunkAccessor& accessor)
{
  return impl->update_mapping(accessor);
}

int PicPoissonPetsc::copy_chunk_to_src(elliptic::ChunkAccessor& accessor)
{
  return impl->copy_chunk_to_src(accessor);
}

int PicPoissonPetsc::copy_sol_to_chunk(elliptic::ChunkAccessor& accessor)
{
  return impl->copy_sol_to_chunk(accessor);
}

int PicPoissonPetsc::set_option(const nlohmann::json& config)
{
  const auto& solver_config = config.value("poisson_petsc", nlohmann::json::object());
  return impl->set_option(solver_config);
}

int PicPoissonPetsc::solve(elliptic::ChunkAccessor& accessor)
{
  return impl->solve(accessor);
}

int PicPoissonPetsc::scatter_forward()
{
  return impl->scatter_forward();
}

int PicPoissonPetsc::scatter_reverse()
{
  return impl->scatter_reverse();
}

int PicPoissonPetsc::scatter_forward_begin()
{
  return impl->scatter_forward_begin();
}

int PicPoissonPetsc::scatter_forward_end()
{
  return impl->scatter_forward_end();
}

int PicPoissonPetsc::scatter_reverse_begin()
{
  return impl->scatter_reverse_begin();
}

int PicPoissonPetsc::scatter_reverse_end()
{
  return impl->scatter_reverse_end();
}

Vec PicPoissonPetsc::get_vector_src_g() const
{
  return impl->get_vector_src_g();
}

Vec PicPoissonPetsc::get_vector_sol_g() const
{
  return impl->get_vector_sol_g();
}
