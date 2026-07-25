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

# --- nlohmann_json ---
if(NOT TARGET nlohmann_json::nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        4f8fba14066156b73f1189a2b8bd568bde5284c5  # v3.10.5
  )
  list(APPEND _picnix_fetch_dependencies nlohmann_json)
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
endif()

# --- xtl ---
if(NOT TARGET xtl)
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        a7c1c5444dfc57f76620391af4c94785ff82c8d6  # 0.7.7
  )
  list(APPEND _picnix_fetch_dependencies xtl)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(DOWNLOAD_GTEST OFF CACHE BOOL "" FORCE)
endif()

# --- xsimd ---
if(NOT TARGET xsimd)
  FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        c1247bffa8fc36de7380a5cd42673a3b32f74c97  # 12.1.1
  )
  list(APPEND _picnix_fetch_dependencies xsimd)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(XSIMD_SKIP_INSTALL ON CACHE BOOL "" FORCE)
endif()

# --- xtensor ---
if(NOT TARGET xtensor)
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        44b56bbae2185ebf19e6f617ac5690344b9e35a4  # 0.24.7
    PATCH_COMMAND  ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P
      "${CMAKE_CURRENT_LIST_DIR}/patches/apply_patch.cmake"
  )
  list(APPEND _picnix_fetch_dependencies xtensor)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(DOWNLOAD_GTEST OFF CACHE BOOL "" FORCE)
endif()

# --- toml11 ---
if(NOT TARGET toml11::toml11)
  FetchContent_Declare(
    toml11
    GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
    GIT_TAG        7c336a52a0100b24a57be811873db9b5fce9f45a  # v4.0.1
  )
  list(APPEND _picnix_fetch_dependencies toml11)
  set(TOML11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(TOML11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()

# --- plog ---
if(NOT TARGET plog::plog)
  FetchContent_Declare(
    plog
    GIT_REPOSITORY https://github.com/SergiusTheBest/plog.git
    GIT_TAG        e21baecd4753f14da64ede979c5a19302618b752  # 1.1.10
  )
  list(APPEND _picnix_fetch_dependencies plog)
  set(PLOG_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(PLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PLOG_INSTALL OFF CACHE BOOL "" FORCE)
endif()

# --- mdspan ---
if(NOT TARGET std::mdspan)
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        9ceface91483775a6c74d06ebf717bbb2768452f  # mdspan-0.6.0
  )
  list(APPEND _picnix_fetch_dependencies mdspan)
  set(MDSPAN_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(MDSPAN_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MDSPAN_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MDSPAN_ENABLE_COMP_BENCH OFF CACHE BOOL "" FORCE)
endif()

# --- fmt ---
if(NOT TARGET fmt::fmt)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        123913715afeb8a437e6388b4473fcc4753e1c9a  # 11.1.4
  )
  list(APPEND _picnix_fetch_dependencies fmt)
  set(FMT_TEST OFF CACHE BOOL "" FORCE)
  set(FMT_DOC OFF CACHE BOOL "" FORCE)
  set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
endif()

# ── populate in dependency order ─────────────────────────────────────
# Declarations above are ordered so xtl and xsimd precede xtensor.
if(_picnix_fetch_dependencies)
  FetchContent_MakeAvailable(
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
