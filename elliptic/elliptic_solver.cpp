#include <utility>

#include "elliptic.hpp"

#if PICNIX_ENABLE_PETSC
#include "petsc_interface.hpp"
#endif

namespace elliptic
{

Solver::Solver(PtrInterface interface) : interface(std::move(interface))
{
}

void Solver::set_interface(PtrInterface interface)
{
  this->interface = std::move(interface);
}

int Solver::update_mapping(ChunkAccessor& accessor)
{
  if (interface == nullptr)
    return 1;
  return interface->update_mapping(accessor);
}

int Solver::copy_chunk_to_src(ChunkAccessor& accessor)
{
  if (interface == nullptr)
    return 1;
  return interface->copy_chunk_to_src(accessor);
}

int Solver::copy_sol_to_chunk(ChunkAccessor& accessor)
{
  if (interface == nullptr)
    return 1;
  return interface->copy_sol_to_chunk(accessor);
}

int Solver::set_option(const nlohmann::json& config)
{
  if (interface == nullptr)
    return 1;
  return interface->set_option(config);
}

int Solver::solve(ChunkAccessor& accessor)
{
  if (interface == nullptr)
    return 1;

  const int expected = accessor.get_num_grids_total();

  int status = update_mapping(accessor);
  if (status != 0)
    return status;

  const int packed = copy_chunk_to_src(accessor);
  if (packed != expected)
    return 1;

  status = scatter_forward();
  if (status != 0)
    return status;

  status = interface->solve(accessor);
  if (status != 0)
    return status;

  status = scatter_reverse();
  if (status != 0)
    return status;

  const int unpacked = copy_sol_to_chunk(accessor);
  return (unpacked == expected) ? 0 : 1;
}

int Solver::scatter_forward()
{
  if (interface == nullptr)
    return 1;
  return interface->scatter_forward();
}

int Solver::scatter_reverse()
{
  if (interface == nullptr)
    return 1;
  return interface->scatter_reverse();
}

Solver::PtrInterface Solver::get_interface()
{
  return interface;
}

std::shared_ptr<const Solver::Interface> Solver::get_interface() const
{
  return interface;
}

int Solver::initialize(int* argc, char*** argv)
{
#if PICNIX_ENABLE_PETSC
  PetscBool      petsc_initialized = PETSC_FALSE;
  PetscErrorCode ierr              = PetscInitialized(&petsc_initialized);
  if (ierr != 0)
    return ierr;

  if (petsc_initialized) {
    petsc_initialized_by_us = false;
    return 0;
  }

  ierr = PetscInitialize(argc, argv, nullptr, nullptr);
  if (ierr != 0)
    return ierr;

  petsc_initialized_by_us = true;
#else
  (void)argc;
  (void)argv;
#endif
  return 0;
}

int Solver::finalize()
{
#if PICNIX_ENABLE_PETSC
  PetscBool      petsc_finalized = PETSC_FALSE;
  PetscErrorCode ierr            = PetscFinalized(&petsc_finalized);
  if (ierr != 0)
    return ierr;

  if (petsc_initialized_by_us && !petsc_finalized) {
    ierr = PetscFinalize();
    if (ierr != 0)
      return ierr;
  }

  petsc_initialized_by_us = false;
#endif
  return 0;
}

} // namespace elliptic
