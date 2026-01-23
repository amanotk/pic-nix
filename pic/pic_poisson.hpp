// -*- C++ -*-
#ifndef _PIC_POISSON_HPP_
#define _PIC_POISSON_HPP_

#include "pic.hpp"
#include "pic_chunk.hpp"

#include "elliptic/chunk_accessor.hpp"
#include "elliptic/petsc_poisson.hpp"
#include <petscvec.h>

class PicPoisson
{
public:
  using ChunkVec = std::vector<PicChunk*>;

  PicPoisson(const nix::Dims3D& global_dims, float64 delh);

  int     update_mapping(const ChunkVec& chunks);
  int     scatter_rhs(const ChunkVec& chunks);
  int     copy_rhs_to_solution();
  int     scatter_solution_to_chunks(const ChunkVec& chunks);
  int     solve(const ChunkVec& chunks);
  int     set_option(const json& config);
  float64 get_residual_norm();

private:
  class PicChunkAccessor : public elliptic::ChunkAccessor
  {
  public:
    PicChunkAccessor() : chunk_size_(0)
    {
    }

    void set_chunks(const ChunkVec& chunks);

    virtual void build_global_index(std::vector<int>& index, nix::Dims3D dims) const override;
    virtual int  pack(float64* buffer, int size) override;
    virtual int  unpack(float64* buffer, int size) override;
    virtual int  get_num_chunks() const override;
    virtual int  get_num_grids_per_chunk() const override;
    virtual int  get_num_grids_total() const override;

  private:
    ChunkVec    chunks_;
    nix::Dims3D chunk_dims_;
    int         chunk_size_;
  };

  struct SolverBase {
    virtual ~SolverBase()                                                = default;
    virtual int     update_mapping(elliptic::ChunkAccessor& accessor)    = 0;
    virtual int     copy_chunk_to_src(elliptic::ChunkAccessor& accessor) = 0;
    virtual int     copy_sol_to_chunk(elliptic::ChunkAccessor& accessor) = 0;
    virtual int     scatter_forward_begin()                              = 0;
    virtual int     scatter_forward_end()                                = 0;
    virtual int     scatter_reverse_begin()                              = 0;
    virtual int     scatter_reverse_end()                                = 0;
    virtual int     solve()                                              = 0;
    virtual int     set_option(const json& config)                       = 0;
    virtual float64 get_residual_norm()                                  = 0;
    virtual int     copy_rhs_to_solution()                               = 0;
  };

  template <typename T>
  class SolverAdapter : public T, public SolverBase
  {
  public:
    using T::T;
    using elliptic::PetscInterface::scatter_forward_begin;
    using elliptic::PetscInterface::scatter_forward_end;
    using elliptic::PetscInterface::scatter_reverse_begin;
    using elliptic::PetscInterface::scatter_reverse_end;

    int update_mapping(elliptic::ChunkAccessor& accessor) override
    {
      return T::update_mapping(accessor);
    }

    int copy_chunk_to_src(elliptic::ChunkAccessor& accessor) override
    {
      return T::copy_chunk_to_src(accessor);
    }

    int copy_sol_to_chunk(elliptic::ChunkAccessor& accessor) override
    {
      return T::copy_sol_to_chunk(accessor);
    }

    int scatter_forward_begin() override
    {
      return T::scatter_forward_begin();
    }

    int scatter_forward_end() override
    {
      return T::scatter_forward_end();
    }

    int scatter_reverse_begin() override
    {
      return T::scatter_reverse_begin();
    }

    int scatter_reverse_end() override
    {
      return T::scatter_reverse_end();
    }

    int solve() override
    {
      return T::solve();
    }

    int set_option(const json& config) override
    {
      return T::set_option(config);
    }

    float64 get_residual_norm() override
    {
      return T::get_residual_norm();
    }

    int copy_rhs_to_solution() override
    {
      return VecCopy(this->vector_src_g, this->vector_sol_g);
    }
  };

  using SolverPtr = std::unique_ptr<SolverBase>;

  nix::Dims3D      global_dims_;
  SolverPtr        solver_;
  PicChunkAccessor accessor_;
};

#endif //_PIC_POISSON_HPP_
