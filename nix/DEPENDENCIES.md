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

| Library | Version | Repository | CMake target |
|---------|---------|------------|--------------|
| Catch2 | 3.5.4 | <https://github.com/catchorg/Catch2> | `Catch2::Catch2` (test only) |
| fmt | 11.1.4 | <https://github.com/fmtlib/fmt> | `fmt::fmt` |
| nlohmann/json | 3.10.5 | <https://github.com/nlohmann/json> | `nlohmann_json::nlohmann_json` |
| toml11 | 4.0.1 | <https://github.com/ToruNiina/toml11> | `toml11::toml11` |
| plog | 1.1.10 | <https://github.com/SergiusTheBest/plog> | `plog::plog` |
| xtl | 0.7.7 | <https://github.com/xtensor-stack/xtl> | `xtl` |
| xsimd | 12.1.1 | <https://github.com/xtensor-stack/xsimd> | `xsimd` |
| xtensor | 0.24.7 | <https://github.com/xtensor-stack/xtensor> | `xtensor` |
| mdspan | 0.6.0 | <https://github.com/kokkos/mdspan> | `std::mdspan` |

xtensor is installed with a patch for LLVM 19 compatibility
(`nix/cmake/patches/xtensor-0.24.7-llvm19.patch`).
