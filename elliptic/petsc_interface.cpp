#include "petsc_interface.hpp"

#include <cassert>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include <petscdmda.h>
#include <petscsys.h>

namespace elliptic
{

PetscInterface::PetscInterface(Dims3D dims)
    : dims(dims), dm_obj(nullptr), ksp_obj(nullptr), matrix(nullptr), vector_src_l(nullptr),
      vector_src_g(nullptr), vector_sol_l(nullptr), vector_sol_g(nullptr)
{
}

PetscInterface::~PetscInterface()
{
  destroy_petsc_objects();
}

void PetscInterface::destroy_petsc_objects()
{
  if (ksp_obj != nullptr) {
    KSPDestroy(&ksp_obj);
  }
  if (matrix != nullptr) {
    MatDestroy(&matrix);
  }
  if (vector_src_l != nullptr) {
    VecDestroy(&vector_src_l);
  }
  if (vector_src_g != nullptr) {
    VecDestroy(&vector_src_g);
  }
  if (vector_sol_l != nullptr) {
    VecDestroy(&vector_sol_l);
  }
  if (vector_sol_g != nullptr) {
    VecDestroy(&vector_sol_g);
  }
  if (dm_obj != nullptr) {
    DMDestroy(&dm_obj);
  }
}

void PetscInterface::initialize()
{
  PetscInitialize(nullptr, nullptr, nullptr, nullptr);
}

void PetscInterface::finalize()
{
  PetscFinalize();
}

std::string PetscInterface::bool_to_string(bool x)
{
  return x ? "true" : "false";
}

std::string PetscInterface::int_to_string(int x)
{
  return std::to_string(x);
}

std::string PetscInterface::float_to_string(double x)
{
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(5) << x;
  return oss.str();
}

int PetscInterface::apply_petsc_option(const OptionVec& opts)
{
  //
  // Note:
  // There does not seem to be a way to check valid PETSc options.
  //
  for (const auto& [key, val] : opts) {
    const std::string opt = "-" + key;

    PetscErrorCode ierr = PetscOptionsSetValue(NULL, opt.c_str(), val.c_str());
    if (ierr != 0) {
      PetscPrintf(PETSC_COMM_WORLD, "failed to set PETSc option: %s\n", opt.c_str());
      return 1;
    }
  }

  return 0;
}

PetscInterface::OptionVec PetscInterface::make_petsc_option(const nlohmann::json& config)
{
  OptionVec option{};

  if (!config.is_object()) {
    PetscPrintf(PETSC_COMM_WORLD, "PETSc configuration should be a JSON object.\n");
    return option;
  }

  for (auto it = config.begin(); it != config.end(); ++it) {
    const std::string key = it.key();
    const auto&       val = it.value();

    if (val.is_null())
      continue;

    if (val.is_boolean()) {
      option.emplace_back(key, bool_to_string(val.get<bool>()));
    }

    if (val.is_number_integer()) {
      option.emplace_back(key, int_to_string(val.get<int>()));
      continue;
    }

    if (val.is_number_float()) {
      option.emplace_back(key, float_to_string(val.get<double>()));
      continue;
    }

    if (val.is_string()) {
      option.emplace_back(key, val.get<std::string>());
      continue;
    }
  }

  return option;
}

PetscInterface::OptionVec PetscInterface::make_petsc_option(const toml::value& config)
{
  OptionVec option{};

  if (!config.is_table()) {
    PetscPrintf(PETSC_COMM_WORLD, "PETSc configuration should be a table.\n");
    return option;
  }

  for (const auto& item : config.as_table()) {
    const std::string key = item.first;
    const auto&       val = item.second;

    if (val.is_boolean()) {
      option.emplace_back(key, bool_to_string(val.as_boolean()));
      continue;
    }

    if (val.is_integer()) {
      option.emplace_back(key, int_to_string(val.as_integer()));
      continue;
    }

    if (val.is_floating()) {
      option.emplace_back(key, float_to_string(val.as_floating()));
      continue;
    }

    if (val.is_string()) {
      option.emplace_back(key, val.as_string());
      continue;
    }
  }

  return option;
}

int PetscInterface::scatter_forward_begin()
{
  scatter->scatter_forward_begin(vector_src_l, vector_src_g);
  return 0;
}

int PetscInterface::scatter_forward_end()
{
  scatter->scatter_forward_end(vector_src_l, vector_src_g);
  return 0;
}

int PetscInterface::scatter_reverse_begin()
{
  scatter->scatter_reverse_begin(vector_sol_l, vector_sol_g);
  return 0;
}

int PetscInterface::scatter_reverse_end()
{
  scatter->scatter_reverse_end(vector_sol_l, vector_sol_g);
  return 0;
}

int PetscInterface::update_mapping(ChunkAccessor& accessor)
{
  scatter->setup_scatter(accessor, src_buf, sol_buf, vector_src_l, vector_sol_l, vector_src_g);
  return 0;
}

int PetscInterface::copy_chunk_to_src(ChunkAccessor& accessor)
{
  return accessor.pack(src_buf.data(), static_cast<int>(src_buf.size()));
}

int PetscInterface::copy_sol_to_chunk(ChunkAccessor& accessor)
{
  return accessor.unpack(sol_buf.data(), static_cast<int>(sol_buf.size()));
}

int PetscInterface::set_option(const nlohmann::json& config)
{
  auto it = config.find("petsc");
  if (it == config.end() || !it->is_object()) {
    return 0;
  }

  return apply_petsc_option(make_petsc_option(*it));
}

void PetscInterface::create_dm(Dims3D dims)
{
  assert(dims.size() == 3);

  if (dm_obj != nullptr) {
    DMDestroy(&dm_obj);
  }

  const bool is_3d   = (dims[0] >= 2) && (dims[1] >= 2) && (dims[2] >= 2);
  const bool is_2d   = (dims[0] == 1) && (dims[1] >= 2) && (dims[2] >= 2);
  const bool is_1d   = (dims[0] == 1) && (dims[1] == 1) && (dims[2] >= 2);
  const bool invalid = (!is_3d) && (!is_2d) && (!is_1d);

  if (invalid) {
    ERROR << "Invalid dimension is specified for PetscInterface::create_dm()";
    return;
  } else if (is_1d) {
    create_dm1d(dims);
  } else if (is_2d) {
    create_dm2d(dims);
  } else {
    create_dm3d(dims);
  }

  DMSetUp(dm_obj);
}

void PetscInterface::create_dm1d(Dims3D dims)
{
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, dims[2], 1, 1, nullptr, &dm_obj);
}

void PetscInterface::create_dm2d(Dims3D dims)
{
  DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
               dims[2], dims[1], PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm_obj);
}

void PetscInterface::create_dm3d(Dims3D dims)
{
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
               DMDA_STENCIL_STAR, dims[2], dims[1], dims[0], PETSC_DECIDE, PETSC_DECIDE,
               PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
}

void PetscInterface::setup()
{
  destroy_petsc_objects();
  create_dm(dims);
  
  if (dm_obj == nullptr) {
    return;
  }

  // create global vectors (local vectors are created in update_mapping)
  DMCreateGlobalVector(dm_obj, &vector_src_g);
  DMCreateGlobalVector(dm_obj, &vector_sol_g);

  // create matrix
  set_matrix();

  // create KSP solver
  KSPCreate(PETSC_COMM_WORLD, &ksp_obj);
  KSPSetOperators(ksp_obj, matrix, matrix);
  KSPSetFromOptions(ksp_obj);

  // scatter object
  scatter = std::make_unique<PetscScatter>(&dm_obj, dims);
}

} // namespace elliptic
