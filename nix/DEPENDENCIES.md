# Dependencies

## Vendored

| Library | License | Upstream | Commit |
|---------|---------|----------|--------|
| `cmdline.hpp` (single-header) | BSD-3-Clause | <https://github.com/tanakh/cmdline> | `e4cd007fb8f0314002d9a5b4d82939106e4144e4` |

## Managed

Dependencies are resolved by `nix/cmake/Dependencies.cmake`.  Two modes:

| Mode | CMake option | Behaviour |
|------|-------------|-----------|
| System | `PICNIX_USE_SYSTEM_LIBS=ON` (default) | Try installed packages first (`find_package`); fall back to FetchContent |
| FetchContent | `PICNIX_USE_SYSTEM_LIBS=OFF` | Always fetch pinned versions from GitHub (requires network) |

## Standalone Build

When `nix/` is built as a self-contained project (not as a subdirectory of pic-nix):

```sh
cmake -S . -B build
```

## Root-Project Build

When `nix/` is consumed by the parent pic-nix repository:

```sh
cmake -S nix -B build-nix
```

## Installing Dependencies to a Custom Prefix (HPC / no-admin)

```sh
scripts/install_dependencies.sh "$HOME/usr"
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/usr"
```

The install script lives under `scripts/` in the pic-nix repository — copy or
symlink it when building `nix` standalone.

### Cross-compilation

Pass the same CMake initial-cache or toolchain option to the dependency
installer and the PIC-NIX build.  This is required for compiled dependencies
such as fmt and Catch2; making them static does not make host-built objects
compatible with a different target architecture.
Relative cache and toolchain paths passed to the installer are resolved from
the directory where the script is invoked.  The installer translates
`--toolchain` to `CMAKE_TOOLCHAIN_FILE` for compatibility with CMake 3.20.

For example, from the pic-nix repository root:

```sh
scripts/install_dependencies.sh "$HOME/usr-fugaku" \
  -C cmake/fugaku-llvm22-cross.cmake
cmake -S . -B build \
  -C cmake/fugaku-llvm22-cross.cmake \
  -DCMAKE_PREFIX_PATH="$HOME/usr-fugaku" \
  -DPICNIX_USE_SYSTEM_LIBS=ON
```

## Offline Build

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/usr" \
  -DPICNIX_USE_SYSTEM_LIBS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

When FetchContent mode is used without network access, CMake’s fetch step
will fail.  Pre-install all dependencies with the install script and use
System mode (`PICNIX_USE_SYSTEM_LIBS=ON`) instead, or ensure a populated
`FETCHCONTENT_BASE_DIR` cache is available.

## Pinned Versions

| Library | Version | Repository | Resolved commit | CMake target |
|---------|---------|------------|-----------------|--------------|
| Catch2 | 3.16.0 | <https://github.com/catchorg/Catch2> | `317ac1ed4c0bb6e6b91eafc817e05c488feffcb3` | `Catch2::Catch2` (test only) |
| fmt | 12.2.0 | <https://github.com/fmtlib/fmt> | `1be298e1bd68957e4cd352e1f676f00e07dcfb57` | `fmt::fmt` |
| nlohmann/json | 3.12.0 | <https://github.com/nlohmann/json> | `55f93686c01528224f448c19128836e7df245f72` | `nlohmann_json::nlohmann_json` |
| toml11 | 4.4.0 | <https://github.com/ToruNiina/toml11> | `be08ba2be2a964edcdb3d3e3ea8d100abc26f286` | `toml11::toml11` |
| plog | 1.1.11 | <https://github.com/SergiusTheBest/plog> | `e5c033e317a01b2703d13aab42288d09b2efdafc` | `plog::plog` |
| xtl | 0.7.7 | <https://github.com/xtensor-stack/xtl> | `a7c1c5444dfc57f76620391af4c94785ff82c8d6` | `xtl` |
| xsimd | 14.3.0 | <https://github.com/xtensor-stack/xsimd> | `e88a72831858123924f7118f345dfe5d70d95991` | `xsimd` |
| xtensor | 0.25.0 | <https://github.com/xtensor-stack/xtensor> | `3634f2ded19e0cf38208c8b86cea9e1d7c8e397d` | `xtensor` |
| mdspan | 0.6.0 | <https://github.com/kokkos/mdspan> | `9ceface91483775a6c74d06ebf717bbb2768452f` | `std::mdspan` |

xtensor is installed with a patch for LLVM 19 compatibility
(`nix/cmake/patches/xtensor-0.25.0-llvm19.patch`).

## Optional ADIOS2

ADIOS2 is an external optional dependency and is not built by
`scripts/install_dependencies.sh`.  The optional nightly CI job builds ADIOS2
2.12.1 with MPI and Python support using `scripts/install_adios2.sh`.  Local
users can build the same installation with:

```sh
python3 -m pip install mpi4py numpy
scripts/install_adios2.sh "$HOME/adios2" --python "$(command -v python3)"
```

## Optional Ascent

Ascent is an external optional dependency and is not built by
`scripts/install_dependencies.sh`.  The optional nightly CI job builds Ascent
0.9.5 with MPI and Python support using `scripts/install_ascent.sh`.  The
installer also builds Ascent's Conduit and visualization dependencies and
creates a Python environment under the installation prefix.  Local users can
build the same installation with:

```sh
scripts/install_ascent.sh "$HOME/ascent" --python "$(command -v python3)"
```

Enable Ascent in the PIC build with `PICNIX_ENABLE_ASCENT=ON` and point
`PICNIX_ASCENT_ROOT` at the installation prefix.  The Ascent CMake package is
located below `<prefix>/ascent-checkout/lib/cmake/ascent`.
