// -*- C++ -*-
#ifndef _PIC_ASCENT_FIELD_SCHEMA_HPP_
#define _PIC_ASCENT_FIELD_SCHEMA_HPP_

#include <array>
#include <string_view>

namespace pic_ascent
{
inline constexpr int raw_schema_version = 1;

struct ComponentLocation {
  std::string_view      name;
  std::string_view      association;
  std::array<double, 3> normalized_xyz;
};

inline constexpr std::array<ComponentLocation, 6> uf_components = {{
    {"Ex", "x-face", {0.0, 0.5, 0.5}},
    {"Ey", "y-face", {0.5, 0.0, 0.5}},
    {"Ez", "z-face", {0.5, 0.5, 0.0}},
    {"Bx", "x-edge", {0.5, 0.0, 0.0}},
    {"By", "y-edge", {0.0, 0.5, 0.0}},
    {"Bz", "z-edge", {0.0, 0.0, 0.5}},
}};

inline constexpr std::array<ComponentLocation, 4> uj_components = {{
    {"rho", "cell", {0.5, 0.5, 0.5}},
    {"Jx", "x-face", {0.0, 0.5, 0.5}},
    {"Jy", "y-face", {0.5, 0.0, 0.5}},
    {"Jz", "z-face", {0.5, 0.5, 0.0}},
}};

inline constexpr std::array<std::string_view, 14> um_components = {{
    "t",
    "x",
    "y",
    "z",
    "tt",
    "xx",
    "yy",
    "zz",
    "tx",
    "ty",
    "tz",
    "xy",
    "yz",
    "zx",
}};
} // namespace pic_ascent

#endif
