#include "petsc_scatter.hpp"

namespace elliptic
{

int PetscScatter::get_indexset(IS& is_obj, std::vector<int>& index)
{
  PetscInt        size;
  const PetscInt* data;

  ISGetLocalSize(is_obj, &size);
  ISGetIndices(is_obj, &data);

  index.resize(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    index[i] = data[i];
  }

  ISRestoreIndices(is_obj, &data);

  return 0;
}

PetscScatter::PetscScatter(DM* dm) : dm_ptr(dm), sc_obj(nullptr), is_obj_l(nullptr), is_obj_g(nullptr)
{
}

PetscScatter::~PetscScatter()
{
  if (sc_obj != nullptr) {
    VecScatterDestroy(&sc_obj);
  }
  if (is_obj_l != nullptr) {
    ISDestroy(&is_obj_l);
  }
  if (is_obj_g != nullptr) {
    ISDestroy(&is_obj_g);
  }
}

int PetscScatter::scatter_forward_begin(Vec& src, Vec& dst)
{
  VecScatterBegin(sc_obj, src, dst, INSERT_VALUES, SCATTER_FORWARD);
  return 0;
}

int PetscScatter::scatter_forward_end(Vec& src, Vec& dst)
{
  VecScatterEnd(sc_obj, src, dst, INSERT_VALUES, SCATTER_FORWARD);
  return 0;
}

int PetscScatter::scatter_reverse_begin(Vec& src, Vec& dst)
{
  VecScatterBegin(sc_obj, dst, src, INSERT_VALUES, SCATTER_REVERSE);
  return 0;
}

int PetscScatter::scatter_reverse_end(Vec& src, Vec& dst)
{
  VecScatterEnd(sc_obj, dst, src, INSERT_VALUES, SCATTER_REVERSE);
  return 0;
}

int PetscScatter::setup_vector_local(std::vector<float64>& buffer, Vec& vec)
{
  if (vec != nullptr) {
    VecDestroy(&vec);
  }

  VecCreateSeqWithArray(PETSC_COMM_SELF, 1, static_cast<PetscInt>(buffer.size()), buffer.data(), &vec);
  return 0;
}

int PetscScatter::setup_indexset_local(int size)
{
  if (is_obj_l != nullptr) {
    ISDestroy(&is_obj_l);
  }

  ISCreateStride(PETSC_COMM_SELF, size, 0, 1, &is_obj_l);
  return 0;
}

int PetscScatter::setup_indexset_global(std::vector<int>& index)
{
  if (is_obj_g != nullptr) {
    ISDestroy(&is_obj_g);
  }

  AO        ao_obj;
  const int size = static_cast<int>(index.size());
  PetscInt* data = index.data();

  DMDAGetAO(*dm_ptr, &ao_obj); // DO NOT destroy it!
  AOApplicationToPetsc(ao_obj, size, data);
  ISCreateGeneral(PETSC_COMM_WORLD, size, data, PETSC_COPY_VALUES, &is_obj_g);

  return 0;
}

int PetscScatter::setup_scatter(Vec& vec_local, Vec& vec_global)
{
  if (sc_obj != nullptr) {
    VecScatterDestroy(&sc_obj);
  }

  VecScatterCreate(vec_local, is_obj_l, vec_global, is_obj_g, &sc_obj);
  return 0;
}

int PetscScatter::get_indexset_local(std::vector<int>& index)
{
  return get_indexset(is_obj_l, index);
}

int PetscScatter::get_indexset_global(std::vector<int>& index)
{
  return get_indexset(is_obj_g, index);
}

} // namespace elliptic
