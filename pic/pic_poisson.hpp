// -*- C++ -*-
#ifndef _PIC_POISSON_HPP_
#define _PIC_POISSON_HPP_

#include "pic.hpp"
#include "pic_chunk.hpp"

#include "elliptic/chunk_accessor.hpp"
#include "elliptic/petsc_interface.hpp"
#include <petscvec.h>

class PicPoisson : public elliptic::PetscInterface
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

protected:
  int set_matrix() override;

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

  nix::Dims3D      global_dims_;
  PicChunkAccessor accessor_;
  float64          delx;
  float64          dely;
  float64          delz;
  void             set_nullspace();
  int              solve() override;
  int              solve(elliptic::ChunkAccessor& accessor) override;
};

#endif //_PIC_POISSON_HPP_
