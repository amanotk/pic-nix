// -*- C++ -*-
#ifndef _PIC_POISSON_BASIC_HPP_
#define _PIC_POISSON_BASIC_HPP_

#include "pic_poisson.hpp"

#include <memory>
#include <nlohmann/json.hpp>

// Basic placeholder Poisson solver until the CG + SSOR implementation is ready.
class PicPoissonBasic : public PicPoisson
{
public:
  PicPoissonBasic(const nix::Dims3D& global_dims, float64 delh);
  ~PicPoissonBasic() override;

  int update_mapping(elliptic::ChunkAccessor& accessor) override;
  int copy_chunk_to_src(elliptic::ChunkAccessor& accessor) override;
  int copy_sol_to_chunk(elliptic::ChunkAccessor& accessor) override;
  int set_option(const nlohmann::json& config) override;
  int solve(elliptic::ChunkAccessor& accessor) override;
  int scatter_forward() override;
  int scatter_reverse() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif //_PIC_POISSON_BASIC_HPP_
