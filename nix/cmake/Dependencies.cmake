cmake_minimum_required(VERSION 3.20)

# ── option ─────────────────────────────────────────────────────────────
if(NOT DEFINED PICNIX_USE_SYSTEM_LIBS)
  option(PICNIX_USE_SYSTEM_LIBS
    "Prefer system-installed libraries over FetchContent" ON)
endif()

# ── system discovery ───────────────────────────────────────────────────
if(PICNIX_USE_SYSTEM_LIBS)
  find_package(fmt 11 CONFIG QUIET)
  find_package(nlohmann_json 3.10 CONFIG QUIET)
  find_package(toml11 4.0 CONFIG QUIET)
  find_package(plog 1.1 CONFIG QUIET)
  find_package(xtl 0.7 CONFIG QUIET)
  find_package(xsimd 12 CONFIG QUIET)
  find_package(xtensor 0.24 CONFIG QUIET)
  find_package(mdspan 0.6 QUIET)
endif()

# xtensor's source build reads these component variables when the dependency
# targets already exist. Installed xtl/xsimd configs expose only the lowercase
# package version variables.
if(TARGET xtl AND DEFINED xtl_VERSION AND NOT DEFINED XTL_VERSION_MAJOR)
  string(REPLACE "." ";" _picnix_xtl_version "${xtl_VERSION}")
  list(GET _picnix_xtl_version 0 XTL_VERSION_MAJOR)
  list(GET _picnix_xtl_version 1 XTL_VERSION_MINOR)
  list(GET _picnix_xtl_version 2 XTL_VERSION_PATCH)
endif()

if(TARGET xsimd AND DEFINED xsimd_VERSION AND NOT DEFINED XSIMD_VERSION_MAJOR)
  string(REPLACE "." ";" _picnix_xsimd_version "${xsimd_VERSION}")
  list(GET _picnix_xsimd_version 0 XSIMD_VERSION_MAJOR)
  list(GET _picnix_xsimd_version 1 XSIMD_VERSION_MINOR)
  list(GET _picnix_xsimd_version 2 XSIMD_VERSION_PATCH)
endif()

# ── FetchContent declarations ──────────────────────────────────────────
include(FetchContent)
set(_picnix_fetch_dependencies)

function(_picnix_make_dependencies_available)
  # Keep third-party build options local so embedding projects are unaffected.
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(JSON_BuildTests OFF)
  set(BUILD_TESTS OFF)
  set(DOWNLOAD_GTEST OFF)
  set(BUILD_BENCHMARK OFF)
  set(BUILD_EXAMPLES OFF)
  set(XSIMD_SKIP_INSTALL ON)
  set(TOML11_BUILD_TESTS OFF)
  set(TOML11_BUILD_EXAMPLES OFF)
  set(PLOG_BUILD_SAMPLES OFF)
  set(PLOG_BUILD_TESTS OFF)
  set(PLOG_INSTALL OFF)
  set(MDSPAN_ENABLE_TESTS OFF)
  set(MDSPAN_ENABLE_EXAMPLES OFF)
  set(MDSPAN_ENABLE_BENCHMARKS OFF)
  set(MDSPAN_ENABLE_COMP_BENCH OFF)
  set(FMT_TEST OFF)
  set(FMT_DOC OFF)
  set(FMT_INSTALL OFF)

  FetchContent_MakeAvailable(${ARGV})
endfunction()

# --- nlohmann_json ---
if(NOT TARGET nlohmann_json::nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        55f93686c01528224f448c19128836e7df245f72  # v3.12.0
  )
  list(APPEND _picnix_fetch_dependencies nlohmann_json)
endif()

# --- xtl ---
if(NOT TARGET xtl)
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        a7c1c5444dfc57f76620391af4c94785ff82c8d6  # 0.7.7
  )
  list(APPEND _picnix_fetch_dependencies xtl)
endif()

# --- xsimd ---
if(NOT TARGET xsimd)
  FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        e88a72831858123924f7118f345dfe5d70d95991  # 14.3.0
  )
  list(APPEND _picnix_fetch_dependencies xsimd)
endif()

# --- xtensor ---
if(NOT TARGET xtensor)
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        3634f2ded19e0cf38208c8b86cea9e1d7c8e397d  # 0.25.0
    PATCH_COMMAND  ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P
      "${CMAKE_CURRENT_LIST_DIR}/patches/apply_patch.cmake"
  )
  list(APPEND _picnix_fetch_dependencies xtensor)
endif()

# --- toml11 ---
if(NOT TARGET toml11::toml11)
  FetchContent_Declare(
    toml11
    GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
    GIT_TAG        be08ba2be2a964edcdb3d3e3ea8d100abc26f286  # v4.4.0
  )
  list(APPEND _picnix_fetch_dependencies toml11)
endif()

# --- plog ---
if(NOT TARGET plog::plog)
  FetchContent_Declare(
    plog
    GIT_REPOSITORY https://github.com/SergiusTheBest/plog.git
    GIT_TAG        e5c033e317a01b2703d13aab42288d09b2efdafc  # 1.1.11
  )
  list(APPEND _picnix_fetch_dependencies plog)
endif()

# --- mdspan ---
if(NOT TARGET std::mdspan)
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        9ceface91483775a6c74d06ebf717bbb2768452f  # mdspan-0.6.0
  )
  list(APPEND _picnix_fetch_dependencies mdspan)
endif()

# --- fmt ---
if(NOT TARGET fmt::fmt)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        1be298e1bd68957e4cd352e1f676f00e07dcfb57  # 12.2.0
  )
  list(APPEND _picnix_fetch_dependencies fmt)
endif()

# ── populate in dependency order ─────────────────────────────────────
# Declarations above are ordered so xtl and xsimd precede xtensor.
if(_picnix_fetch_dependencies)
  _picnix_make_dependencies_available(
    ${_picnix_fetch_dependencies}
  )
endif()

# ── aggregate INTERFACE target ─────────────────────────────────────────
if(NOT TARGET picnix_dependencies)
  add_library(picnix_dependencies INTERFACE)
  add_library(picnix::dependencies ALIAS picnix_dependencies)

  target_link_libraries(picnix_dependencies INTERFACE
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    plog::plog
    xtl
    xsimd
    xtensor
  )
  target_compile_definitions(picnix_dependencies INTERFACE XTENSOR_USE_XSIMD)
endif()
