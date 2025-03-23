// -*- C++ -*-
#ifndef _ENGINE_HPP_
#define _ENGINE_HPP_

#include "nix/nix.hpp"

#include "nix/esirkepov.hpp"
#include "nix/interp.hpp"
#include "nix/primitives.hpp"

#include "engine/current.hpp"
#include "engine/maxwell.hpp"
#include "engine/moment.hpp"
#include "engine/position.hpp"
#include "engine/velocity.hpp"

namespace engine
{

constexpr int  DimensionSize    = 3;
constexpr int  OrderSize        = 4;
constexpr auto supported_orders = std::integer_sequence<int, 1, 2, 3, 4>{};

template <typename T_chunk>
class Maxwell
{
public:
  using T_data = typename T_chunk::data_type;

  void init_friedman(T_chunk& chunk, const T_data& data) const
  {
    BaseMaxwell maxwell(data);
    maxwell.init_friedman(data.uf, data.ff);
  }

  auto get_diverror(int dimension, T_chunk& chunk, const T_data& data) const
  {
    BaseMaxwell maxwell(data);
    float64     efderr = 0.0;
    float64     bfderr = 0.0;

    int xbuffer[2] = {0, 0};
    int ybuffer[2] = {0, 0};
    int zbuffer[2] = {0, 0};

    //
    // avoid error calculation around physical boundary
    //
    if (chunk.get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      xbuffer[0] = data.boundary_margin;
    }

    if (chunk.get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      xbuffer[1] = data.boundary_margin;
    }

    if (chunk.get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      ybuffer[0] = data.boundary_margin;
    }

    if (chunk.get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      ybuffer[1] = data.boundary_margin;
    }

    if (chunk.get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
      zbuffer[0] = data.boundary_margin;
    }

    if (chunk.get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
      zbuffer[1] = data.boundary_margin;
    }

    //
    // calculate divergence error
    //
    if (dimension == 1) {
      maxwell.get_diverror_1d(data.uf, data.uj, efderr, bfderr, xbuffer);
    }

    if (dimension == 2) {
      maxwell.get_diverror_2d(data.uf, data.uj, efderr, bfderr, xbuffer, ybuffer);
    }

    if (dimension == 3) {
      maxwell.get_diverror_3d(data.uf, data.uj, efderr, bfderr, xbuffer, ybuffer, zbuffer);
    }

    return std::make_pair(efderr, bfderr);
  }

  void push_efd(int dimension, T_chunk& chunk, const T_data& data, float64 delt) const
  {
    BaseMaxwell maxwell(data);

    if (dimension == 1) {
      maxwell.push_efd_1d(data.uf, data.uj, data.ff, delt);
    }

    if (dimension == 2) {
      maxwell.push_efd_2d(data.uf, data.uj, data.ff, delt);
    }

    if (dimension == 3) {
      maxwell.push_efd_3d(data.uf, data.uj, data.ff, delt);
    }
  }

  void push_bfd(int dimension, T_chunk& chunk, const T_data& data, float64 delt) const
  {
    BaseMaxwell maxwell(data);

    if (dimension == 1) {
      maxwell.push_bfd_1d(data.uf, data.uj, data.ff, delt);
    }

    if (dimension == 2) {
      maxwell.push_bfd_2d(data.uf, data.uj, data.ff, delt);
    }

    if (dimension == 3) {
      maxwell.push_bfd_3d(data.uf, data.uj, data.ff, delt);
    }
  }
};

template <typename T_chunk>
class Current
{
private:
  static constexpr int size_table = 2 * DimensionSize * OrderSize;
  using T_data                    = typename T_chunk::data_type;
  using func_ptr_t                = void (*)(T_chunk&, const T_data&, double);
  using func_table_t              = std::array<func_ptr_t, size_table>;

  static constexpr int encode(int is_vector, int dimension, int order)
  {
    constexpr auto stride0 = DimensionSize * OrderSize;
    constexpr auto stride1 = OrderSize;
    constexpr auto stride2 = 1;

    return is_vector * stride0 + (dimension - 1) * stride1 + (order - 1) * stride2;
  }

  template <int isVector, int Dim, int Order>
  static void call_entry(T_chunk& chunk, const T_data& data, double delt)
  {
    bool has_dim[3] = {chunk.has_z_dim(), chunk.has_y_dim(), chunk.has_x_dim()};

    if constexpr (isVector == 0) {
      ScalarCurrent<Dim, Order> current(data, has_dim);
      current(data.up, data.uj, delt);
    }

    if constexpr (isVector == 1) {
      VectorCurrent<Dim, Order> current(data, has_dim);
      current(data.up, data.uj, delt);
    }
  }

  template <int... Orders>
  static void fill_table_entries(func_table_t& table, std::integer_sequence<int, Orders...> orders)
  {
    // scalar and 1D
    ((table[encode(0, 1, Orders)] = &call_entry<0, 1, Orders>), ...);
    // scalar and 2D
    ((table[encode(0, 2, Orders)] = &call_entry<0, 2, Orders>), ...);
    // scalar and 3D
    ((table[encode(0, 3, Orders)] = &call_entry<0, 3, Orders>), ...);
    // vector and 1D
    ((table[encode(1, 1, Orders)] = &call_entry<1, 1, Orders>), ...);
    // vector and 2D
    ((table[encode(1, 2, Orders)] = &call_entry<1, 2, Orders>), ...);
    // vector and 3D
    ((table[encode(1, 3, Orders)] = &call_entry<1, 3, Orders>), ...);
  }

  static func_table_t create_table_impl()
  {
    func_table_t table = {};

    fill_table_entries(table, supported_orders);

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int dimension, int order, T_chunk& chunk, const T_data& data,
                  float64 delt) const
  {
    table[encode(is_vector, dimension, order)](chunk, data, delt);
  }
};

template <typename T_chunk>
class Moment
{
private:
  static constexpr int size_table = 2 * DimensionSize * OrderSize;
  using T_data                    = typename T_chunk::data_type;
  using func_ptr_t                = void (*)(T_chunk&, const T_data&);
  using func_table_t              = std::array<func_ptr_t, size_table>;

  static constexpr int encode(int is_vector, int dimension, int order)
  {
    constexpr auto stride0 = DimensionSize * OrderSize;
    constexpr auto stride1 = OrderSize;
    constexpr auto stride2 = 1;

    return is_vector * stride0 + (dimension - 1) * stride1 + (order - 1) * stride2;
  }

  template <int isVector, int Dim, int Order>
  static void call_entry(T_chunk& chunk, const T_data& data)
  {
    bool has_dim[3] = {chunk.has_z_dim(), chunk.has_y_dim(), chunk.has_x_dim()};

    if constexpr (isVector == 0) {
      ScalarMoment<Dim, Order> moment(data, has_dim);
      moment(data.up, data.um);
    }

    if constexpr (isVector == 1) {
      VectorMoment<Dim, Order> moment(data, has_dim);
      moment(data.up, data.um);
    }
  }

  template <int... Orders>
  static void fill_table_entries(func_table_t& table, std::integer_sequence<int, Orders...> orders)
  {
    // scalar and 1D
    ((table[encode(0, 1, Orders)] = &call_entry<0, 1, Orders>), ...);
    // scalar and 2D
    ((table[encode(0, 2, Orders)] = &call_entry<0, 2, Orders>), ...);
    // scalar and 3D
    ((table[encode(0, 3, Orders)] = &call_entry<0, 3, Orders>), ...);
    // vector and 1D
    ((table[encode(1, 1, Orders)] = &call_entry<1, 1, Orders>), ...);
    // vector and 2D
    ((table[encode(1, 2, Orders)] = &call_entry<1, 2, Orders>), ...);
    // vector and 3D
    ((table[encode(1, 3, Orders)] = &call_entry<1, 3, Orders>), ...);
  }

  static func_table_t create_table_impl()
  {
    func_table_t table = {};

    fill_table_entries(table, supported_orders);

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int dimension, int order, T_chunk& chunk, const T_data& data) const
  {
    table[encode(is_vector, dimension, order)](chunk, data);
  }
};

template <typename T_chunk>
class Position
{
private:
  static constexpr int size_table = 2;
  using T_data                    = typename T_chunk::data_type;
  using func_ptr_t                = void (*)(T_chunk&, const T_data&, int, double);
  using func_table_t              = std::array<func_ptr_t, size_table>;

  template <int isVector>
  static void call_entry(T_chunk& chunk, const T_data& data, int order, double delt)
  {
    if constexpr (isVector == 0) {
      ScalarPosition position(data);
      position(data.up, delt);
      set_boundary(position, chunk, data, order, delt);
    }

    if constexpr (isVector == 1) {
      VectorPosition position(data);
      position(data.up, delt);
      set_boundary(position, chunk, data, order, delt);
    }
  }

  template <typename T_position>
  static void set_boundary(T_position& position, T_chunk& chunk, const T_data& data, int order,
                           double delt)
  {
    bool has_dim[3] = {chunk.has_z_dim(), chunk.has_y_dim(), chunk.has_x_dim()};

    for (int is = 0; is < data.Ns; is++) {
      chunk.set_boundary_particle(data.up[is], 0, data.up[is]->Np - 1, is);
      position.count(data.up[is], 0, data.up[is]->Np - 1, true, order, has_dim);
    }
  }

  static func_table_t create_table_impl()
  {
    func_table_t table = {};

    // scalar
    table[0] = &call_entry<0>;
    // vector
    table[1] = &call_entry<1>;

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int order, T_chunk& chunk, const T_data& data, double delt) const
  {
    table[is_vector](chunk, data, order, delt);
  }

  template <typename T_particle>
  void count(int order, T_chunk& chunk, const T_data& data, T_particle particle, int Lbp, int Ubp,
             bool reset) const
  {
    bool has_dim[3] = {chunk.has_z_dim(), chunk.has_y_dim(), chunk.has_x_dim()};

    BasePosition position(data);
    position.count(particle, Lbp, Ubp, reset, order, has_dim);
  }
};

template <typename T_chunk>
class Velocity
{
private:
  static constexpr int size_table = 2 * DimensionSize * OrderSize * PusherSize * InterpSize;
  using T_data                    = typename T_chunk::data_type;
  using func_ptr_t                = void (*)(T_chunk&, const T_data&, double);
  using func_table_t              = std::array<func_ptr_t, size_table>;

  static constexpr int encode(int is_vector, int dimension, int order, int pusher, int interp)
  {
    constexpr auto stride0 = DimensionSize * OrderSize * PusherSize * InterpSize;
    constexpr auto stride1 = OrderSize * PusherSize * InterpSize;
    constexpr auto stride2 = PusherSize * InterpSize;
    constexpr auto stride3 = InterpSize;
    constexpr auto stride4 = 1;

    return is_vector * stride0 + (dimension - 1) * stride1 + (order - 1) * stride2 +
           pusher * stride3 + interp * stride4;
  }

  template <int isVector, int Dim, int Order, int Pusher, int Interp>
  static void call_entry(T_chunk& chunk, const T_data& data, double delt)
  {
    bool has_dim[3] = {chunk.has_z_dim(), chunk.has_y_dim(), chunk.has_x_dim()};

    if constexpr (isVector == 0) {
      ScalarVelocity<Dim, Order, Pusher, Interp> velocity(data, has_dim);
      velocity(data.up, data.uf, delt);
    }

    if constexpr (isVector == 1) {
      VectorVelocity<Dim, Order, Pusher, Interp> velocity(data, has_dim);
      velocity(data.up, data.uf, delt);
    }
  }

  template <int Pusher, int Interp, int... Orders>
  static void fill_table_entries(func_table_t& table, std::integer_sequence<int, Orders...> orders)
  {
    // scalar and 1D
    ((table[encode(0, 1, Orders, Pusher, Interp)] = &call_entry<0, 1, Orders, Pusher, Interp>),
     ...);
    // scalar and 2D
    ((table[encode(0, 2, Orders, Pusher, Interp)] = &call_entry<0, 2, Orders, Pusher, Interp>),
     ...);
    // scalar and 3D
    ((table[encode(0, 3, Orders, Pusher, Interp)] = &call_entry<0, 3, Orders, Pusher, Interp>),
     ...);
    // vector and 1D
    ((table[encode(1, 1, Orders, Pusher, Interp)] = &call_entry<1, 1, Orders, Pusher, Interp>),
     ...);
    // vector and 2D
    ((table[encode(1, 2, Orders, Pusher, Interp)] = &call_entry<1, 2, Orders, Pusher, Interp>),
     ...);
    // vector and 3D
    ((table[encode(1, 3, Orders, Pusher, Interp)] = &call_entry<1, 3, Orders, Pusher, Interp>),
     ...);
  }

  static func_table_t create_table_impl()
  {
    constexpr auto orders = std::integer_sequence<int, 1, 2, 3, 4>{};
    func_table_t   table  = {};

    // Boris
    fill_table_entries<PusherBoris, InterpMC>(table, orders);
    fill_table_entries<PusherBoris, InterpWT>(table, orders);
    // Vay
    fill_table_entries<PusherVay, InterpMC>(table, orders);
    fill_table_entries<PusherVay, InterpWT>(table, orders);
    // Higuera-Cary
    fill_table_entries<PusherHigueraCary, InterpMC>(table, orders);
    fill_table_entries<PusherHigueraCary, InterpWT>(table, orders);

    return table;
  }

  /// initialize the function table
  inline static const func_table_t table = create_table_impl();

public:
  void operator()(int is_vector, int dimension, int order, int pusher, int interp, T_chunk& chunk,
                  const T_data& data, float64 delt) const
  {
    table[encode(is_vector, dimension, order, pusher, interp)](chunk, data, delt);
  }
};

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif