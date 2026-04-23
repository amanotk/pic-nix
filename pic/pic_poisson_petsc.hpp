// -*- C++ -*-
#ifndef _PIC_POISSON_PETSC_HPP_
#define _PIC_POISSON_PETSC_HPP_

#include "pic_poisson.hpp"

#include <memory>
#include <petscvec.h>

class PicPoissonPetsc : public PicPoisson
{
public:
  PicPoissonPetsc(const nix::Dims3D& global_dims, float64 delh);
  ~PicPoissonPetsc() override;

  int update_mapping(elliptic::ChunkAccessor& accessor) override;
  int copy_chunk_to_src(elliptic::ChunkAccessor& accessor) override;
  int copy_sol_to_chunk(elliptic::ChunkAccessor& accessor) override;
  int set_option(const nlohmann::json& config) override;
  int solve(elliptic::ChunkAccessor& accessor) override;
  int scatter_forward() override;
  int scatter_reverse() override;
  int scatter_forward_begin();
  int scatter_forward_end();
  int scatter_reverse_begin();
  int scatter_reverse_end();
  Vec get_vector_src_g() const;
  Vec get_vector_sol_g() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

#endif //_PIC_POISSON_PETSC_HPP_
