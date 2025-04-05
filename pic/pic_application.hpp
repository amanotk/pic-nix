// -*- C++ -*-
#ifndef _PIC_APPLICATION_HPP_
#define _PIC_APPLICATION_HPP_

#include "nix/application.hpp"

#include "pic.hpp"

class PicApplicationInterface : public nix::Application::Interface
{
public:
  virtual PtrChunk create_chunk(const int dims[], const bool has_dim[], int id) = 0;

  virtual int get_num_species();

  virtual void calculate_moment();
};

///
/// @brief Application for 3D Explicit PIC Simulations
///
/// This class takes care of initialization of various objects (ChunkMap, Balancer, etc.) and a main
/// loop for explicit PIC simulations. Core numerical solvers are implemented in a chunk class, and
/// this class only deals with the management of a collection of chunks.
///
/// A problem specific application class must be defined by deriving this class, which should
/// override the virtual factory method create_chunk() to return a pointer to a problem specific
/// chunk instance. In addition, custom diagnostics routines may also be implemented through the
/// virtual method diagnostic().
///
class PicApplication : public nix::Application
{
public:
  using this_type  = PicApplication;
  using base_type  = nix::Application;
  using MpiCommVec = xt::xtensor_fixed<MPI_Comm, xt::xshape<NumBoundaryMode, 3, 3, 3>>;

  PicApplication(int argc, char** argv, PtrInterface interface);

  virtual int get_num_species() const;

  virtual void calculate_moment();

protected:
  int        Ns;         ///< number of species
  int        momstep;    ///< step at which moment quantities are cached
  MpiCommVec mpicommvec; ///< MPI Communicators

  virtual void initialize(int argc, char** argv) override;

  virtual void initialize_diagnostic() override;

  virtual void set_chunk_communicator();

  virtual void setup_chunks() override;

  virtual bool rebalance() override;

  virtual void finalize() override;

  virtual std::string get_basedir() override;

  virtual json to_json() override;

  virtual bool from_json(json& state) override;

  virtual void push() override;

  virtual void push_openmp();

  virtual void calculate_moment_openmp();

  virtual void push_taskflow();

  virtual void calculate_moment_taskflow();
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
