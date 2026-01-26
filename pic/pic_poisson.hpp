// -*- C++ -*-
#ifndef _PIC_POISSON_HPP_
#define _PIC_POISSON_HPP_

#include "pic.hpp"
#include "pic_chunk.hpp"

#include "elliptic/chunk_accessor.hpp"
#include "elliptic/petsc_interface.hpp"
#include <memory>
#include <petscvec.h>

class PicPoisson : public elliptic::PetscInterface
{
public:
  using ChunkVec = std::vector<std::unique_ptr<PicChunk>>;

  PicPoisson(const nix::Dims3D& global_dims, float64 delh);

  class PicChunkAccessor : public elliptic::ChunkAccessor
  {
  public:
    explicit PicChunkAccessor(const ChunkVec& chunks);

    virtual void build_global_index(std::vector<int>& index, nix::Dims3D dims) const override;
    virtual int  pack(float64* buffer, int size) override;
    virtual int  unpack(float64* buffer, int size) override;
    virtual int  get_num_chunks() const override;
    virtual int  get_num_grids_per_chunk() const override;
    virtual int  get_num_grids_total() const override;

  private:
    const ChunkVec& chunks_;
    nix::Dims3D     chunk_dims_;
    int             chunk_size_;
  };
  int set_option(const json& config);
  int solve(elliptic::ChunkAccessor& accessor) override;

protected:
  int set_matrix() override;

private:
  nix::Dims3D global_dims_;
  float64     delx;
  float64     dely;
  float64     delz;
  void        set_nullspace();
};

#endif //_PIC_POISSON_HPP_
