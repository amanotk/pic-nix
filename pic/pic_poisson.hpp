// -*- C++ -*-
#ifndef _PIC_POISSON_HPP_
#define _PIC_POISSON_HPP_

#include "pic.hpp"
#include "pic_chunk.hpp"

#include "nix/application.hpp"

#include "elliptic/chunk_accessor.hpp"
#include "elliptic/petsc_interface.hpp"
#include <memory>
#include <petscvec.h>

class PicPoisson : public elliptic::PetscInterface
{
public:
  using AppChunkVec  = nix::Application::ChunkVec;
  using PicChunkVec  = std::vector<std::unique_ptr<PicChunk>>;
  using ChunkViewVec = std::vector<PicChunk*>;

  class PicChunkAccessor : public elliptic::ChunkAccessor
  {
  public:
    PicChunkAccessor(const ChunkViewVec& chunks, nix::Dims3D chunk_dims);

    virtual void build_global_index(std::vector<int>& index, nix::Dims3D dims) const override;
    virtual int  pack(float64* buffer, int size) override;
    virtual int  unpack(float64* buffer, int size) override;
    virtual int  get_num_chunks() const override;
    virtual int  get_num_grids_per_chunk() const override;
    virtual int  get_num_grids_total() const override;

  private:
    const ChunkViewVec& chunkvec;
    nix::Dims3D         chunk_dims;
  };

  PicPoisson(const nix::Dims3D& global_dims, float64 delh);
  int              set_option(const json& config);
  int              solve(elliptic::ChunkAccessor& accessor) override;
  void             bind_chunks(AppChunkVec& chunkvec);
  void             bind_chunks(PicChunkVec& chunkvec);
  PicChunkAccessor get_accessor();

protected:
  template <typename ChunkContainer>
  void bind_chunks_impl(ChunkContainer& chunkvec);

  nix::Dims3D  global_dims;
  float64      delx;
  float64      dely;
  float64      delz;
  ChunkViewVec chunk_views;
  nix::Dims3D  chunk_dims;
  void         set_nullspace() override;
  int          set_matrix() override;
};

template <typename ChunkContainer>
void PicPoisson::bind_chunks_impl(ChunkContainer& chunkvec)
{
  chunk_views.clear();
  chunk_dims = {0, 0, 0};

  if (chunkvec.size() == 0) {
    return;
  }

  chunk_views.reserve(chunkvec.size());
  for (auto& chunk_ptr : chunkvec) {
    auto* pic_chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    if (pic_chunk == nullptr) {
      ERROR << "PicPoisson requires PicChunk-compatible chunks." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    chunk_views.push_back(pic_chunk);
  }

  auto dims  = chunk_views.front()->get_dims();
  chunk_dims = {dims[0], dims[1], dims[2]};
}

#endif //_PIC_POISSON_HPP_
