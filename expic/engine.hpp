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

class BaseEngine
{
public:
  static constexpr int size_table = 64;

  static constexpr int encode(int is_vector, int dimension, int order)
  {
    return is_vector * 32 + (dimension - 1) * 8 + (order - 1);
  }
};

template <typename T_data>
class CurrentEngine : public BaseEngine
{
public:
  using func_ptr_t   = void (*)(const T_data&, double);
  using func_table_t = std::array<func_ptr_t, size_table>;

private:
  template <int isVector, int Dim, int Order>
  static void call_entry(const T_data& data, double delt)
  {
    if constexpr (isVector == 0) {
      ScalarCurrent<Dim, Order> current(data);
      current(data.up, data.uj, delt);
    }

    if constexpr (isVector == 1) {
      VectorCurrent<Dim, Order> current(data);
      current(data.up, data.uj, delt);
    }
  }

  static func_table_t create_table_impl()
  {
    func_table_t table = {};

    // scalar and 3D
    table[encode(0, 3, 1)] = &call_entry<0, 3, 1>;
    table[encode(0, 3, 2)] = &call_entry<0, 3, 2>;
    table[encode(0, 3, 3)] = &call_entry<0, 3, 3>;
    table[encode(0, 3, 4)] = &call_entry<0, 3, 4>;
    // vector and 3D
    table[encode(1, 3, 1)] = &call_entry<1, 3, 1>;
    table[encode(1, 3, 2)] = &call_entry<1, 3, 2>;
    table[encode(1, 3, 3)] = &call_entry<1, 3, 3>;
    table[encode(1, 3, 4)] = &call_entry<1, 3, 4>;

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int dimension, int order, const T_data& data, float64 delt) const
  {
    table[encode(is_vector, dimension, order)](data, delt);
  }
};

template <typename T_data>
class MomentEngine : public BaseEngine
{
public:
  using func_ptr_t   = void (*)(const T_data&);
  using func_table_t = std::array<func_ptr_t, size_table>;

private:
  template <int isVector, int Dim, int Order>
  static void call_entry(const T_data& data)
  {
    if constexpr (isVector == 0) {
      ScalarMoment<Dim, Order> moment(data);
      moment(data.up, data.um);
    }

    if constexpr (isVector == 1) {
      VectorMoment<Dim, Order> moment(data);
      moment(data.up, data.um);
    }
  }

  static func_table_t create_table_impl()
  {
    func_table_t table = {};

    // scalar and 3D
    table[encode(0, 3, 1)] = &call_entry<0, 3, 1>;
    table[encode(0, 3, 2)] = &call_entry<0, 3, 2>;
    table[encode(0, 3, 3)] = &call_entry<0, 3, 3>;
    table[encode(0, 3, 4)] = &call_entry<0, 3, 4>;
    // vector and 3D
    table[encode(1, 3, 1)] = &call_entry<1, 3, 1>;
    table[encode(1, 3, 2)] = &call_entry<1, 3, 2>;
    table[encode(1, 3, 3)] = &call_entry<1, 3, 3>;
    table[encode(1, 3, 4)] = &call_entry<1, 3, 4>;

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int dimension, int order, const T_data& data) const
  {
    table[encode(is_vector, dimension, order)](data);
  }
};

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif