// -*- C++ -*-
#include "pic_poisson_factory.hpp"

#include "pic_poisson_basic.hpp"

#if PICNIX_ENABLE_PETSC
#include "pic_poisson_petsc.hpp"
#endif

std::shared_ptr<PicPoisson> make_poisson_solver(const nix::Dims3D& global_dims, float64 delh)
{
#if PICNIX_ENABLE_PETSC
  return std::make_shared<PicPoissonPetsc>(global_dims, delh);
#else
  return std::make_shared<PicPoissonBasic>(global_dims, delh);
#endif
}
