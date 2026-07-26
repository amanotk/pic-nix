// -*- C++ -*-
#ifndef _HYBRID_HPP_
#define _HYBRID_HPP_

namespace hybrid
{
inline constexpr int boundary_margin = 2;
inline constexpr int particle_order  = 2;

inline constexpr int num_fluid_components         = 10;
inline constexpr int num_field_components         = 6;
inline constexpr int num_moment_components        = 10;
inline constexpr int num_current_components       = 4;
inline constexpr int num_vector_components        = 3;
inline constexpr int num_phase_directions         = 3;
inline constexpr int num_phase_branches           = 3;
inline constexpr int num_conserved_components     = 5;
inline constexpr int num_reconstructed_components = 16;
inline constexpr int num_ohm_source_components    = 4;

namespace fluid_component
{
inline constexpr int electron_density    = 0;
inline constexpr int electron_velocity_x = 1;
inline constexpr int electron_velocity_y = 2;
inline constexpr int electron_velocity_z = 3;
inline constexpr int electron_pressure   = 4;
inline constexpr int ion_density         = 5;
inline constexpr int ion_velocity_x      = 6;
inline constexpr int ion_velocity_y      = 7;
inline constexpr int ion_velocity_z      = 8;
inline constexpr int ion_pressure        = 9;
} // namespace fluid_component

namespace field_component
{
inline constexpr int electric_x = 0;
inline constexpr int electric_y = 1;
inline constexpr int electric_z = 2;
inline constexpr int magnetic_x = 3;
inline constexpr int magnetic_y = 4;
inline constexpr int magnetic_z = 5;
} // namespace field_component

namespace moment_component
{
inline constexpr int density    = 0;
inline constexpr int momentum_x = 1;
inline constexpr int momentum_y = 2;
inline constexpr int momentum_z = 3;
inline constexpr int stress_xx  = 4;
inline constexpr int stress_yy  = 5;
inline constexpr int stress_zz  = 6;
inline constexpr int stress_xy  = 7;
inline constexpr int stress_xz  = 8;
inline constexpr int stress_yz  = 9;
} // namespace moment_component

namespace current_component
{
inline constexpr int charge    = 0;
inline constexpr int current_x = 1;
inline constexpr int current_y = 2;
inline constexpr int current_z = 3;
} // namespace current_component

enum LoadMode {
  LoadCell,
  LoadParticle,
  NumLoadModes,
};

enum BoundaryMode {
  BoundaryCopy10,
  BoundaryCopy6,
  BoundaryCopy3,
  BoundaryMomentAccum,
  BoundaryMomentCopy,
  BoundaryParticle,
  NumBoundaryModes,
};
} // namespace hybrid

#endif
