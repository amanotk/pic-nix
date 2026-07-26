#!/usr/bin/env bash
set -euo pipefail

ASCENT_VERSION="0.9.5"
CMAKE_MIN_VERSION="3.23"
CMAKE_BOOTSTRAP_VERSION="3.31.10"

usage() {
  cat <<'EOF'
Usage: scripts/install_ascent.sh [install_prefix] --python <python_executable>

Build Ascent, Conduit, and Ascent's visualization dependencies with MPI and
Python support. The default installation prefix is "$HOME/usr".

The Python interpreter is required explicitly. The installer creates a virtual
environment under <install_prefix>/python-venv and builds the Conduit and
Ascent Python modules for that environment.

Examples:

  scripts/install_ascent.sh --python /path/to/python3
  scripts/install_ascent.sh ./thirdparty --python /path/to/python3

Set MPICC and MPICXX to select MPI compiler wrappers. Set
CMAKE_BUILD_PARALLEL_LEVEL to control parallel build jobs (default: nproc).
EOF
}

PREFIX="$HOME/usr"
PYTHON_EXECUTABLE=""

if (( $# > 0 )) && [[ "$1" != -* ]]; then
  PREFIX="$1"
  shift
fi

while (( $# > 0 )); do
  case "$1" in
    --python)
      if (( $# < 2 )); then
        echo "Missing executable after --python" >&2
        exit 2
      fi
      PYTHON_EXECUTABLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_EXECUTABLE" ]]; then
  echo "--python <python_executable> is required" >&2
  usage >&2
  exit 2
fi

if [[ "$PREFIX" != /* ]]; then
  PREFIX="$PWD/$PREFIX"
fi

if [[ "$PYTHON_EXECUTABLE" != */* ]]; then
  PYTHON_EXECUTABLE="$(command -v "$PYTHON_EXECUTABLE" || true)"
elif [[ "$PYTHON_EXECUTABLE" != /* ]]; then
  PYTHON_EXECUTABLE="$PWD/$PYTHON_EXECUTABLE"
fi

if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Python interpreter is not executable: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

for command in curl git patch tar; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 2
  fi
done

MPICC_EXECUTABLE="${MPICC:-mpicc}"
MPICXX_EXECUTABLE="${MPICXX:-mpicxx}"
for compiler in "$MPICC_EXECUTABLE" "$MPICXX_EXECUTABLE"; do
  if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "MPI compiler wrapper not found: $compiler" >&2
    exit 2
  fi
done

if ! "$PYTHON_EXECUTABLE" -m venv --help >/dev/null 2>&1; then
  echo "Python interpreter does not provide the venv module: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || printf '4')}"
BUILDDIR="$(mktemp -d -t picnix-ascent-XXXX)"
trap 'rm -rf "$BUILDDIR"' EXIT

cmake_is_supported() {
  local version_output major minor
  command -v cmake >/dev/null 2>&1 || return 1
  version_output="$(cmake --version)"
  [[ "$version_output" =~ cmake\ version\ ([0-9]+)\.([0-9]+) ]] || return 1
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  (( major > 3 || (major == 3 && minor >= 23) ))
}

if ! cmake_is_supported; then
  echo "CMake $CMAKE_MIN_VERSION or newer is required; bootstrapping CMake $CMAKE_BOOTSTRAP_VERSION"
  "$PYTHON_EXECUTABLE" -m venv "$BUILDDIR/cmake-venv"
  "$BUILDDIR/cmake-venv/bin/python" -m pip install \
    "cmake==$CMAKE_BOOTSTRAP_VERSION"
  export PATH="$BUILDDIR/cmake-venv/bin:$PATH"
fi

mkdir -p "$PREFIX"

echo "--- Installing Ascent ($ASCENT_VERSION) ---"
echo "Install prefix: $PREFIX"
echo "Python: $PYTHON_EXECUTABLE"
echo "MPI C compiler: $MPICC_EXECUTABLE"
echo "MPI C++ compiler: $MPICXX_EXECUTABLE"

ASCENT_DIR="$BUILDDIR/ascent"
git clone https://github.com/Alpine-DAV/ascent.git "$ASCENT_DIR" \
  --branch "v$ASCENT_VERSION" --depth 1 --recurse-submodules \
  --shallow-submodules

env \
  prefix="$BUILDDIR/work" \
  install_dir="$PREFIX" \
  python_exe="$PYTHON_EXECUTABLE" \
  mpicc_exe="$MPICC_EXECUTABLE" \
  mpicxx_exe="$MPICXX_EXECUTABLE" \
  build_jobs="$BUILD_JOBS" \
  build_pyvenv=true \
  enable_python=ON \
  enable_mpi=ON \
  enable_mpicc=ON \
  enable_openmp=ON \
  enable_fortran=OFF \
  enable_tests=OFF \
  "$ASCENT_DIR/scripts/build_ascent/build_ascent.sh"

cat <<EOF

Ascent $ASCENT_VERSION and its dependencies installed to $PREFIX.

Ascent CMake package:

  $PREFIX/ascent-checkout

Python environment:

  $PREFIX/python-venv
EOF
