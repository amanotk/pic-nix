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

  int       status = update_mapping(accessor);
  const int packed = copy_chunk_to_src(accessor);
  status |= (packed == expected) ? 0 : 1;
  status |= scatter_forward();
  status |= interface->solve(accessor);
  status |= scatter_reverse();
  const int unpacked = copy_sol_to_chunk(accessor);
  status |= (unpacked == expected) ? 0 : 1;
  return status;
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

int Solver::initialize()
{
  if (!is_initialized) {
    is_initialized = true;
    is_finalized   = false;
  }
  return 0;
}

int Solver::finalize()
{
  if (is_initialized && !is_finalized) {
    is_finalized   = true;
    is_initialized = false;
  }
  return 0;
}

} // namespace elliptic
