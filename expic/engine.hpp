// -*- C++ -*-
#ifndef _ENGINE_HPP_
#define _ENGINE_HPP_

#include "nix/nix.hpp"

#include "nix/esirkepov.hpp"
#include "nix/interp.hpp"
#include "nix/primitives.hpp"

#include "engine/current.hpp"
#include "engine/moment.hpp"
#include "engine/position.hpp"
#include "engine/velocity.hpp"

namespace engine
{
// alias
template <int Dim, int Order>
using ScalarVelocityBorisMC = ScalarVelocity<Dim, Order, PusherBoris, ShapeMC>;
template <int Dim, int Order>
using ScalarVelocityBorisWT = ScalarVelocity<Dim, Order, PusherBoris, ShapeWT>;
template <int Dim, int Order>
using VectorVelocityBorisMC = VectorVelocity<Dim, Order, PusherBoris, ShapeMC>;
template <int Dim, int Order>
using VectorVelocityBorisWT = VectorVelocity<Dim, Order, PusherBoris, ShapeWT>;

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif