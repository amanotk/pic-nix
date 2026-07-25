# External Dependencies

This directory previously held vendored third-party header libraries.
They are now managed through CMake (FetchContent + system package discovery).

The only remaining vendored file is:

- **cmdline.hpp** — Simple command line parser (single header, no CMake).
  Upstream: https://github.com/tanakh/cmdline
  Commit: e4cd007fb8f0314002d9a5b4d82939106e4144e4

## Dependency Management

Dependencies are resolved by `nix/cmake/Dependencies.cmake`.  Two modes:

| Mode | CMake option | Behaviour |
|------|-------------|-----------|
| System | `PICNIX_USE_SYSTEM_LIBS=ON` (default) | Try installed packages first, FetchContent fallback |
| FetchContent | `PICNIX_USE_SYSTEM_LIBS=OFF` | Always fetch from GitHub (no network fallback) |

## Installing Dependencies to a Custom Prefix (HPC / no-admin)

```sh
scripts/install_dependencies.sh "$HOME/usr"
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/usr" ...
```

## Offline Build

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/usr" \
  -DPICNIX_USE_SYSTEM_LIBS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

## Pinned Versions

| Library | Version | CMake target |
|---------|---------|-------------|
| fmt | 11.1.4 | `fmt::fmt` |
| nlohmann/json | 3.10.5 | `nlohmann_json::nlohmann_json` |
| toml11 | 4.0.1 | `toml11::toml11` |
| plog | 1.1.10 | `plog::plog` |
| xtl | 0.7.7 | `xtl` |
| xsimd | 12.1.1 | `xsimd` |
| xtensor | 0.24.7 | `xtensor` |
| mdspan | 0.6.0 | `std::mdspan` |
| Catch2 | 3.5.4 | `Catch2::Catch2` (test only) |

xtensor is installed with a patch for LLVM 19 compatibility
(`nix/cmake/patches/xtensor-0.24.7-llvm19.patch`).

## Standalone Submodule Builds

Each submodule (nix, pic, elliptic) supports standalone builds:

```sh
cmake nix/ -B build-nix
cmake pic/ -B build-pic -DPICNIX_DIR=.
cmake elliptic/ -B build-elliptic
```

In those modes `nix/cmake/Dependencies.cmake` (or equivalent for Catch2)
is included automatically.
