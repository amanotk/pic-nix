#!/usr/bin/env bash
set -euo pipefail

ADIOS2_VERSION="2.12.1"
ADIOS2_REPOSITORY="https://github.com/ornladios/ADIOS2.git"

usage() {
  cat <<'EOF'
Usage: scripts/install_adios2.sh [install_prefix] (--python <python_executable> | --no-python) [cmake_options...]

Build ADIOS2 with MPI support. The default installation prefix is "$HOME/usr".

With --python, Python bindings are also built; the interpreter must already
have mpi4py and numpy installed because ADIOS2 uses their headers and runtime
support. With --no-python, only the C++ library is built (analysis can use a
separate Python package such as pip's "adios2").

Additional CMake options are forwarded to the ADIOS2 configuration. Pass the
same initial-cache file used to configure PIC-NIX so compiler flags match
(for example -cxx=icpx with cmake/linux-intel.cmake):

  scripts/install_adios2.sh ./thirdparty/adios2 --no-python \
    -C cmake/linux-intel.cmake

Relative -C and toolchain paths are resolved from the directory where this
script is invoked.

Set MPICC and MPICXX to select MPI compiler wrappers. Set
CMAKE_BUILD_PARALLEL_LEVEL to control parallel build jobs (default: 4;
raise it if you have memory headroom).
EOF
}

absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$PWD" "$1"
  fi
}

PREFIX="$HOME/usr"
PYTHON_EXECUTABLE=""
PYTHON_MODE=""
CMAKE_CONFIGURE_ARGS=()

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
      PYTHON_MODE="on"
      shift 2
      ;;
    --no-python)
      PYTHON_EXECUTABLE=""
      PYTHON_MODE="off"
      shift
      ;;
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CMAKE_CONFIGURE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PYTHON_MODE" ]]; then
  echo "One of --python <python_executable> or --no-python is required" >&2
  usage >&2
  exit 2
fi

if [[ "$PREFIX" != /* ]]; then
  PREFIX="$PWD/$PREFIX"
fi

if [[ "$PYTHON_MODE" == "on" ]]; then
  if [[ "$PYTHON_EXECUTABLE" != */* ]]; then
    PYTHON_EXECUTABLE="$(command -v "$PYTHON_EXECUTABLE" || true)"
  elif [[ "$PYTHON_EXECUTABLE" != /* ]]; then
    PYTHON_EXECUTABLE="$PWD/$PYTHON_EXECUTABLE"
  fi

  if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    echo "Python interpreter is not executable: $PYTHON_EXECUTABLE" >&2
    exit 2
  fi
fi

for command in cmake git; do
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

if [[ "$PYTHON_MODE" == "on" ]]; then
  if ! "$PYTHON_EXECUTABLE" -c 'import mpi4py, numpy' >/dev/null 2>&1; then
    echo "mpi4py and numpy are required in the selected Python environment: $PYTHON_EXECUTABLE" >&2
    exit 2
  fi
fi

BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
BUILDDIR="$(mktemp -d -t picnix-adios2-XXXX)"
trap 'rm -rf "$BUILDDIR"' EXIT

mkdir -p "$PREFIX"

echo "--- Installing ADIOS2 ($ADIOS2_VERSION) ---"
echo "Install prefix: $PREFIX"
if [[ "$PYTHON_MODE" == "on" ]]; then
  echo "Python: $PYTHON_EXECUTABLE"
else
  echo "Python: disabled"
fi
echo "MPI C compiler: $MPICC_EXECUTABLE"
echo "MPI C++ compiler: $MPICXX_EXECUTABLE"
if (( ${#CMAKE_CONFIGURE_ARGS[@]} > 0 )); then
  echo "Extra CMake args: ${CMAKE_CONFIGURE_ARGS[*]}"
fi

ADIOS2_DIR="$BUILDDIR/adios2"
git clone "$ADIOS2_REPOSITORY" "$ADIOS2_DIR" \
  --branch "v$ADIOS2_VERSION" --depth 1

ADIOS2_PYTHON_ARGS=()
if [[ "$PYTHON_MODE" == "on" ]]; then
  ADIOS2_PYTHON_ARGS+=(
    -DADIOS2_USE_Python=ON
    "-DPython_EXECUTABLE=$PYTHON_EXECUTABLE"
  )
else
  ADIOS2_PYTHON_ARGS+=(-DADIOS2_USE_Python=OFF)
fi

cmake -S "$ADIOS2_DIR" -B "$ADIOS2_DIR/build" \
  "${CMAKE_CONFIGURE_ARGS[@]+"${CMAKE_CONFIGURE_ARGS[@]}"}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$MPICC_EXECUTABLE" \
  -DCMAKE_CXX_COMPILER="$MPICXX_EXECUTABLE" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DADIOS2_BUILD_EXAMPLES=OFF \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_CUDA=OFF \
  -DADIOS2_USE_Kokkos=OFF \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_HDF5_VOL=OFF \
  -DADIOS2_USE_SST=OFF \
  -DADIOS2_USE_DataMan=OFF \
  -DADIOS2_USE_DataSpaces=OFF \
  -DADIOS2_USE_Blosc2=OFF \
  -DADIOS2_USE_BZip2=OFF \
  -DADIOS2_USE_ZFP=OFF \
  -DADIOS2_USE_SZ=OFF \
  -DADIOS2_USE_SZ3=OFF \
  -DADIOS2_USE_MGARD=OFF \
  -DADIOS2_USE_LIBPRESSIO=OFF \
  -DADIOS2_USE_PIP=OFF \
  "${ADIOS2_PYTHON_ARGS[@]}"

cmake --build "$ADIOS2_DIR/build" --parallel "$BUILD_JOBS"
cmake --install "$ADIOS2_DIR/build"

ADIOS2_CMAKE_DIR=""
ADIOS2_LIB_DIR=""
for candidate in "$PREFIX/lib/cmake/adios2" "$PREFIX/lib64/cmake/adios2"; do
  if [[ -d "$candidate" ]]; then
    ADIOS2_CMAKE_DIR="$candidate"
    ADIOS2_LIB_DIR="${candidate%/cmake/adios2}"
    break
  fi
done

if [[ -z "$ADIOS2_CMAKE_DIR" ]]; then
  echo "ADIOS2 CMake package was not installed under $PREFIX" >&2
  exit 1
fi

if [[ "$PYTHON_MODE" == "on" ]]; then
  PYTHON_VERSION="$(
    "$PYTHON_EXECUTABLE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'
  )"
  PYTHON_SITE_PACKAGES="$ADIOS2_LIB_DIR/python${PYTHON_VERSION}/site-packages"
  if [[ ! -d "$PYTHON_SITE_PACKAGES/adios2" ]]; then
    echo "ADIOS2 Python module was not installed under $PYTHON_SITE_PACKAGES" >&2
    exit 1
  fi

  PYTHONPATH="$PYTHON_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH="$ADIOS2_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$PYTHON_EXECUTABLE" - <<'PY'
import adios2
import mpi4py

print("adios2:", adios2.__file__)
print("mpi4py:", mpi4py.__file__)
PY
fi

cat <<EOF

ADIOS2 $ADIOS2_VERSION installed to $PREFIX.

CMake package directory:

  $ADIOS2_CMAKE_DIR
EOF

if [[ "$PYTHON_MODE" == "on" ]]; then
  cat <<EOF

Python environment:

  $PYTHON_SITE_PACKAGES
EOF
else
  cat <<'EOF'

Python bindings: disabled (--no-python).
Install a Python reader for analysis, for example:

  uv pip install "adios2>=2.11"
EOF
fi
