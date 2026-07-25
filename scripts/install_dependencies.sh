#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_dependencies.sh [install_prefix] [cmake_options...]

Install all PIC-NIX C++ dependencies (including Catch2) into the given prefix.
Default prefix is "$HOME/usr".

After installation, configure the project with:

  cmake -S . -B build -DCMAKE_PREFIX_PATH=<install_prefix> ...

Additional CMake options are forwarded to every dependency configuration.
For cross-compilation, pass the same initial-cache or toolchain option used
to configure PIC-NIX, for example:

  scripts/install_dependencies.sh "$HOME/usr-fugaku" \
    -C cmake/fugaku-llvm22-cross.cmake

Relative initial-cache and toolchain paths are resolved from the directory
where this script is invoked.

Set CMAKE_BUILD_PARALLEL_LEVEL to control parallel build jobs (default: nproc).
EOF
}

if [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]; then
  usage
  exit 0
fi

PREFIX="${1:-$HOME/usr}"
if (( $# > 0 )); then
  shift
fi

absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$PWD" "$1"
  fi
}

CMAKE_CONFIGURE_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    -C)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      CMAKE_CONFIGURE_ARGS+=("$1" "$(absolute_path "$2")")
      shift 2
      ;;
    --toolchain)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      CMAKE_CONFIGURE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$(absolute_path "$2")")
      shift 2
      ;;
    --toolchain=*)
      CMAKE_CONFIGURE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$(absolute_path "${1#*=}")")
      shift
      ;;
    -DCMAKE_TOOLCHAIN_FILE=*)
      CMAKE_CONFIGURE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$(absolute_path "${1#*=}")")
      shift
      ;;
    *)
      CMAKE_CONFIGURE_ARGS+=("$1")
      shift
      ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="$(mktemp -d -t picnix-deps-XXXX)"
trap 'rm -rf "$BUILDDIR"' EXIT

mkdir -p "$PREFIX"

install_dependency() {
  local name="$1" url="$2" tag="$3"
  shift 3
  local extra_args=("$@")
  local dir="$BUILDDIR/$name"

  echo "--- Installing $name ($tag) ---"
  git clone "$url" "$dir" --branch "$tag" --depth 1
  cmake -S "$dir" -B "$dir/build" \
    "${CMAKE_CONFIGURE_ARGS[@]}" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    "${extra_args[@]}"
  cmake --build "$dir/build" \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"
  cmake --install "$dir/build"
  rm -rf "$dir"
}

# ── fmt (compiled) ──────────────────────────────────────────────────────
echo "--- Installing fmt (11.1.4) ---"
FMT_DIR="$BUILDDIR/fmt"
git clone https://github.com/fmtlib/fmt.git "$FMT_DIR" --branch 11.1.4 --depth 1
cmake -S "$FMT_DIR" -B "$FMT_DIR/build" \
  "${CMAKE_CONFIGURE_ARGS[@]}" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DFMT_TEST=OFF -DFMT_DOC=OFF
cmake --build "$FMT_DIR/build" \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"
cmake --install "$FMT_DIR/build"
rm -rf "$FMT_DIR"

# ── nlohmann/json ───────────────────────────────────────────────────────
install_dependency nlohmann_json \
  https://github.com/nlohmann/json.git v3.10.5 \
  -DJSON_BuildTests=OFF

# ── xtl ─────────────────────────────────────────────────────────────────
install_dependency xtl \
  https://github.com/xtensor-stack/xtl.git 0.7.7 \
  -DBUILD_TESTS=OFF

# ── xsimd ───────────────────────────────────────────────────────────────
install_dependency xsimd \
  https://github.com/xtensor-stack/xsimd.git 12.1.1 \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF

# ── xtensor (with LLVM 19 patch) ────────────────────────────────────────
echo "--- Installing xtensor (0.24.7 + LLVM 19 patch) ---"
XTENSOR_DIR="$BUILDDIR/xtensor"
git clone https://github.com/xtensor-stack/xtensor.git "$XTENSOR_DIR" --branch 0.24.7 --depth 1
git -C "$XTENSOR_DIR" apply "$REPO_ROOT/nix/cmake/patches/xtensor-0.24.7-llvm19.patch"
echo "  LLVM 19 patch applied"
cmake -S "$XTENSOR_DIR" -B "$XTENSOR_DIR/build" \
  "${CMAKE_CONFIGURE_ARGS[@]}" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARK=OFF -DDOWNLOAD_GTEST=OFF
cmake --build "$XTENSOR_DIR/build" \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"
cmake --install "$XTENSOR_DIR/build"
rm -rf "$XTENSOR_DIR"

# ── toml11 ──────────────────────────────────────────────────────────────
install_dependency toml11 \
  https://github.com/ToruNiina/toml11.git v4.0.1 \
  -DTOML11_BUILD_TESTS=OFF -DTOML11_BUILD_EXAMPLES=OFF

# ── plog ────────────────────────────────────────────────────────────────
install_dependency plog \
  https://github.com/SergiusTheBest/plog.git 1.1.10 \
  -DPLOG_BUILD_SAMPLES=OFF -DPLOG_BUILD_TESTS=OFF

# ── mdspan ──────────────────────────────────────────────────────────────
install_dependency mdspan \
  https://github.com/kokkos/mdspan.git mdspan-0.6.0 \
  -DMDSPAN_ENABLE_TESTS=OFF -DMDSPAN_ENABLE_EXAMPLES=OFF \
  -DMDSPAN_ENABLE_BENCHMARKS=OFF -DMDSPAN_CXX_STANDARD=17

# ── Catch2 ───────────────────────────────────────────────────────────────
install_dependency Catch2 \
  https://github.com/catchorg/Catch2.git v3.5.4 \
  -DBUILD_TESTING=OFF -DCATCH_BUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF

# ── done ────────────────────────────────────────────────────────────────
cat <<EOF

All dependencies (including Catch2) installed to $PREFIX.

Configure pic-nix with:

  cmake -S . -B build -DCMAKE_PREFIX_PATH=$PREFIX \\
    -DPICNIX_USE_SYSTEM_LIBS=ON

For a strict offline build (no network access):

  cmake -S . -B build -DCMAKE_PREFIX_PATH=$PREFIX \\
    -DPICNIX_USE_SYSTEM_LIBS=ON \\
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON

For FetchContent-based build (no pre-install needed):

  cmake -S . -B build -DPICNIX_USE_SYSTEM_LIBS=OFF

EOF
