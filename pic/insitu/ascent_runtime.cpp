// -*- C++ -*-
#include "ascent_runtime.hpp"

#include <ascent/ascent_actions_utils.hpp>

#include <stdexcept>

namespace picnix::insitu
{
void AscentRuntime::open(const std::filesystem::path& actions_path)
{
  if (is_open) {
    return;
  }

  if (MPI_Comm_dup(MPI_COMM_WORLD, &comm) != MPI_SUCCESS) {
    throw std::runtime_error("failed to duplicate MPI communicator for Ascent");
  }

  try {
    conduit::Node options;
    options["mpi_comm"] = MPI_Comm_c2f(comm);
    runtime.open(options);

    if (!ascent::load_actions_file(actions_path.string(), MPI_Comm_c2f(comm), actions)) {
      runtime.close();
      MPI_Comm_free(&comm);
      comm = MPI_COMM_NULL;
      throw std::runtime_error("failed to load Ascent actions file: " + actions_path.string());
    }
  } catch (...) {
    if (comm != MPI_COMM_NULL) {
      MPI_Comm_free(&comm);
      comm = MPI_COMM_NULL;
    }
    throw;
  }

  is_open = true;
}

void AscentRuntime::publish_execute(const conduit::Node&         data,
                                    const std::filesystem::path& actions_path)
{
  open(actions_path);
  runtime.publish(data);
  runtime.execute(actions);
}

void AscentRuntime::shutdown()
{
  if (!is_open) {
    return;
  }

  runtime.close();
  if (comm != MPI_COMM_NULL) {
    MPI_Comm_free(&comm);
    comm = MPI_COMM_NULL;
  }
  is_open = false;
  actions.reset();
}
} // namespace picnix::insitu
