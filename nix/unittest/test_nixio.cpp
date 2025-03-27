// -*- C++ -*-

#include "nixio.hpp"
#include <iostream>

#include "catch.hpp"

static const auto json_to_check = R"(
{
  "scalar_int32": {
    "datatype": "i4",
    "description": "",
    "offset": 0,
    "size": 4,
    "ndim": 1,
    "shape": [
      1
    ],
    "data": 32
  },
  "scalar_int64": {
    "datatype": "i8",
    "description": "",
    "offset": 4,
    "size": 8,
    "ndim": 1,
    "shape": [
      1
    ],
    "data": 64
  },
  "array_int32": {
    "datatype": "i4",
    "description": "",
    "offset": 12,
    "size": 20,
    "ndim": 1,
    "shape": [
      5
    ],
    "data": [
      10,
      20,
      30,
      40,
      50
    ]
  },
  "array_int64": {
    "datatype": "i8",
    "description": "",
    "offset": 32,
    "size": 40,
    "ndim": 1,
    "shape": [
      5
    ],
    "data": [
      15,
      30,
      45,
      60,
      75
    ]
  },
  "scalar_float32": {
    "datatype": "f4",
    "description": "",
    "offset": 72,
    "size": 4,
    "ndim": 1,
    "shape": [
      1
    ],
    "data": 1.0
  },
  "scalar_float64": {
    "datatype": "f8",
    "description": "",
    "offset": 76,
    "size": 8,
    "ndim": 1,
    "shape": [
      1
    ],
    "data": 10.0
  },
  "array_float32": {
    "datatype": "f4",
    "description": "",
    "offset": 84,
    "size": 20,
    "ndim": 1,
    "shape": [
      5
    ],
    "data": [
      0.0,
      1.0,
      2.0,
      3.0,
      4.0
    ]
  },
  "array_float64": {
    "datatype": "f8",
    "description": "",
    "offset": 104,
    "size": 40,
    "ndim": 1,
    "shape": [
      5
    ],
    "data": [
      0.0,
      -1.0,
      -2.0,
      -3.0,
      -4.0
    ]
  }
}
)"_json;

template <typename T>
bool is_array_equal(const int N, T x[], T y[])
{
  return std::equal(&x[0], &x[N - 1], y);
}

//
// put/get attribute
//
TEST_CASE("Attribute")
{
  using json = nixio::json;
  using nixio::float32;
  using nixio::float64;

  const int N       = 5;
  int32_t   i32s    = 32;
  int64_t   i64s    = 64;
  int32_t   i32a[N] = {10, 20, 30, 40, 50};
  int64_t   i64a[N] = {15, 30, 45, 60, 75};
  float32   f32s    = 1.0;
  float64   f64s    = 10.0;
  float32   f32a[N] = {0.0, 1.0, 2.0, 3.0, 4.0};
  float64   f64a[N] = {0.0, -1.0, -2.0, -3.0, -4.0};

  //
  // check put_attribute
  //
  {
    size_t disp;
    json   writer, diff;

    disp = 0;
    nixio::put_attribute(writer, "scalar_int32", disp, i32s);
    disp += sizeof(int32_t);

    nixio::put_attribute(writer, "scalar_int64", disp, i64s);
    disp += sizeof(int64_t);

    nixio::put_attribute(writer, "array_int32", disp, N, i32a);
    disp += sizeof(int32_t) * N;

    nixio::put_attribute(writer, "array_int64", disp, N, i64a);
    disp += sizeof(int64_t) * N;

    nixio::put_attribute(writer, "scalar_float32", disp, f32s);
    disp += sizeof(float32);

    nixio::put_attribute(writer, "scalar_float64", disp, f64s);
    disp += sizeof(float64);

    nixio::put_attribute(writer, "array_float32", disp, N, f32a);
    disp += sizeof(float32) * N;

    nixio::put_attribute(writer, "array_float64", disp, N, f64a);
    disp += sizeof(float64) * N;

    diff = json::diff(writer, json_to_check);
    REQUIRE(diff.size() == 0);
    REQUIRE(diff[0].is_null());
  }

  //
  // check get_attribute
  //
  {
    json reader = json_to_check;

    size_t  disp1, disp2;
    int32_t in_i32s;
    int64_t in_i64s;
    int32_t in_i32a[N];
    int64_t in_i64a[N];
    float32 in_f32s;
    float64 in_f64s;
    float32 in_f32a[N];
    float64 in_f64a[N];

    disp1 = 0;
    nixio::get_attribute(reader, "scalar_int32", disp2, in_i32s);
    REQUIRE(disp1 == disp2);
    REQUIRE(i32s == in_i32s);
    disp1 += sizeof(int32_t);

    nixio::get_attribute(reader, "scalar_int64", disp2, in_i64s);
    REQUIRE(disp1 == disp2);
    REQUIRE(i64s == in_i64s);
    disp1 += sizeof(int64_t);

    nixio::get_attribute(reader, "array_int32", disp2, N, in_i32a);
    REQUIRE(disp1 == disp2);
    REQUIRE(is_array_equal(N, i32a, in_i32a));
    disp1 += sizeof(int32_t) * N;

    nixio::get_attribute(reader, "array_int64", disp2, N, in_i64a);
    REQUIRE(disp1 == disp2);
    REQUIRE(is_array_equal(N, i64a, in_i64a));
    disp1 += sizeof(int64_t) * N;

    nixio::get_attribute(reader, "scalar_float32", disp2, in_f32s);
    REQUIRE(disp1 == disp2);
    REQUIRE(f32s == in_f32s);
    disp1 += sizeof(float32);

    nixio::get_attribute(reader, "scalar_float64", disp2, in_f64s);
    REQUIRE(disp1 == disp2);
    REQUIRE(f64s == in_f64s);
    disp1 += sizeof(float64);

    nixio::get_attribute(reader, "array_float32", disp2, N, in_f32a);
    REQUIRE(disp1 == disp2);
    REQUIRE(is_array_equal(N, f32a, in_f32a));
    disp1 += sizeof(float32) * N;

    nixio::get_attribute(reader, "array_float64", disp2, N, in_f64a);
    REQUIRE(disp1 == disp2);
    REQUIRE(is_array_equal(N, f64a, in_f64a));
    disp1 += sizeof(float64) * N;
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
