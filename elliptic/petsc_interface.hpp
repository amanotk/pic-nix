#ifndef _PETSC_INTERFACE_HPP_
#define _PETSC_INTERFACE_HPP_

#include "nix.hpp"
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "elliptic.hpp"
#include "petsc_scatter.hpp"

namespace elliptic
{

using namespace nix::typedefs;

/// @brief PETSc solver interface class
class PetscInterface : public SolverInterface
{
public:
  using Option     = std::pair<std::string, std::string>;
  using OptionVec  = std::vector<Option>;
  using PtrScatter = std::unique_ptr<PetscScatter>;

  static void initialize();
  static void finalize();

  explicit PetscInterface(Dims3D dims);
  virtual ~PetscInterface();

protected:
  Dims3D               dims;
  std::vector<float64> src_local;
  std::vector<float64> sol_local;
  DM                   dm_obj;
  KSP                  ksp_obj;
  Mat                  matrix;
  Vec                  vector_src_l;
  Vec                  vector_src_g;
  Vec                  vector_sol_l;
  Vec                  vector_sol_g;
  OptionVec            option;
  PtrScatter           scatter;

  static std::string bool_to_string(bool x);
  static std::string int_to_string(int x);
  static std::string float_to_string(double x);
  static int         apply_petsc_option(const OptionVec& opts);
  static OptionVec   make_petsc_option(const nlohmann::json& config);
  static OptionVec   make_petsc_option(const toml::value& config);

  virtual int  scatter_forward_begin();
  virtual int  scatter_forward_end();
  virtual int  scatter_reverse_begin();
  virtual int  scatter_reverse_end();
  virtual int  update_mapping(ChunkAccessor& accessor);
  virtual int  copy_chunk_to_src(ChunkAccessor& accessor);
  virtual int  copy_sol_to_chunk(ChunkAccessor& accessor);
  virtual void create_dm(Dims3D dims);
  virtual void create_dm1d(Dims3D dims);
  virtual void create_dm2d(Dims3D dims);
  virtual void create_dm3d(Dims3D dims);
  virtual void set_matrix(float64 hx, float64 hy, float64 hz) = 0;
};

} // namespace elliptic

#endif