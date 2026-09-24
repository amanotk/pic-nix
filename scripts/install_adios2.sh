#!/usr/bin/env bash
set -euo pipefail

ADIOS2_VERSION="2.12.1"
ADIOS2_REPOSITORY="https://github.com/ornladios/ADIOS2.git"

usage() {
  cat <<'EOF'
Usage: scripts/install_adios2.sh [install_prefix] (--python <python_executable> | --no-python) [--cross-build] [--cross-python --target-python <aarch64_python_prefix> --target-numpy <aarch64_numpy_prefix> --target-mpi4py <aarch64_mpi4py_prefix>] [cmake_options...]

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

For a Linux cross-build, --cross-build adds the build-tree shared-library
linker path needed by ADIOS2 utilities. Pass a real CMake toolchain file and
any target-specific try_run results separately.

For cross-compiled Python bindings, use --cross-python with --python
(host x86_64 venv for pip/build steps) and --target-python/--target-numpy/
--target-mpi4py (aarch64 Python 3.11, NumPy, and mpi4py prefixes for headers
and module paths). The host Python runs pip and CMake configure-time scripts;
extensions compile against the target Python ABI.
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
CROSS_BUILD=false
CROSS_PYTHON=false
TARGET_PYTHON=""
TARGET_NUMPY=""
TARGET_MPI4PY=""
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
    --cross-build)
      CROSS_BUILD=true
      shift
      ;;
    --cross-python)
      CROSS_PYTHON=true
      shift
      ;;
    --target-python|--target-numpy|--target-mpi4py)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      case "$1" in
        --target-python) TARGET_PYTHON="$2" ;;
        --target-numpy) TARGET_NUMPY="$2" ;;
        --target-mpi4py) TARGET_MPI4PY="$2" ;;
      esac
      shift 2
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

if [[ "$CROSS_PYTHON" == true && "$PYTHON_MODE" == "off" ]]; then
  echo "--cross-python cannot be combined with --no-python" >&2
  exit 2
fi

if [[ "$CROSS_PYTHON" == true && ( -z "$TARGET_PYTHON" || -z "$TARGET_NUMPY" || -z "$TARGET_MPI4PY" ) ]]; then
  echo "--cross-python requires --target-python, --target-numpy, and --target-mpi4py" >&2
  exit 2
fi

if [[ "$CROSS_BUILD" == true && "$PYTHON_MODE" != "off" && "$CROSS_PYTHON" != true ]]; then
  echo "--cross-build with Python requires --cross-python" >&2
  exit 2
fi

if [[ "$PREFIX" != /* ]]; then
  PREFIX="$PWD/$PREFIX"
fi

if [[ "$CROSS_PYTHON" == true ]]; then
  for variable in TARGET_PYTHON TARGET_NUMPY TARGET_MPI4PY; do
    if [[ "${!variable}" != /* ]]; then
      printf -v "$variable" '%s/%s' "$PWD" "${!variable}"
    fi
  done
  if [[ ! -f "$TARGET_PYTHON/include/python3.11/Python.h" ]]; then
    echo "Target Python 3.11 headers not found under $TARGET_PYTHON" >&2
    exit 2
  fi
  if [[ ! -f "$TARGET_PYTHON/lib/libpython3.11.so" ]]; then
    echo "Target Python library not found under $TARGET_PYTHON" >&2
    exit 2
  fi
  if [[ ! -f "$TARGET_NUMPY/lib/python3.11/site-packages/numpy/core/include/numpy/arrayobject.h" ]]; then
    echo "Target NumPy headers not found under $TARGET_NUMPY" >&2
    exit 2
  fi
  if [[ ! -f "$TARGET_MPI4PY/lib/python3.11/site-packages/mpi4py/include/mpi4py/mpi4py.h" ]]; then
    echo "Target mpi4py headers not found under $TARGET_MPI4PY" >&2
    exit 2
  fi
  # ADIOS2 needs mpi4py at configure time and nanobind derives the extension
  # suffix from Python_SOABI, both of which come from the host interpreter
  # unless overridden. Read the target values from its sysconfig data.
  TARGET_SYSCONFIG=""
  for candidate in "$TARGET_PYTHON"/lib/python3.11/_sysconfigdata__*.py; do
    if [[ -f "$candidate" ]]; then
      TARGET_SYSCONFIG="$candidate"
      break
    fi
  done
  if [[ -z "$TARGET_SYSCONFIG" ]]; then
    echo "Target Python sysconfig data not found under $TARGET_PYTHON" >&2
    exit 2
  fi
  TARGET_SOABI="$(grep -m1 "^ *'SOABI':" "$TARGET_SYSCONFIG" | cut -d"'" -f4 || true)"
  if [[ -z "$TARGET_SOABI" ]]; then
    echo "Could not determine target Python SOABI from $TARGET_SYSCONFIG" >&2
    exit 2
  fi
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
MPICC_EXECUTABLE="$(command -v "$MPICC_EXECUTABLE")"
MPICXX_EXECUTABLE="$(command -v "$MPICXX_EXECUTABLE")"

if [[ "$PYTHON_MODE" == "on" && "$CROSS_PYTHON" != true ]]; then
  if ! "$PYTHON_EXECUTABLE" -c 'import mpi4py, numpy' >/dev/null 2>&1; then
    echo "mpi4py and numpy are required in the selected Python environment: $PYTHON_EXECUTABLE" >&2
    exit 2
  fi
fi

if [[ "$CROSS_PYTHON" == true ]]; then
  if ! "$PYTHON_EXECUTABLE" -c 'import pip, numpy; assert numpy.__version__ == "1.26.4"; import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
    echo "Cross-build host Python must be 3.11 with pip and NumPy 1.26.4: $PYTHON_EXECUTABLE" >&2
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

ADIOS2_CROSS_ARGS=()
if [[ "$CROSS_BUILD" == true ]]; then
  ADIOS2_CROSS_ARGS+=(
    -DCMAKE_INSTALL_LIBDIR=lib
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$ADIOS2_DIR/build/lib"
  )
fi

ADIOS2_PYTHON_ARGS=()
if [[ "$CROSS_PYTHON" == true ]]; then
  ADIOS2_PYTHON_ARGS+=(
    -DADIOS2_USE_Python=ON
    "-DPython_EXECUTABLE=$PYTHON_EXECUTABLE"
    "-DPython_INCLUDE_DIR=$TARGET_PYTHON/include/python3.11"
    "-DPython_LIBRARY=$TARGET_PYTHON/lib/libpython3.11.so"
    "-DPython3_INCLUDE_DIR=$TARGET_PYTHON/include/python3.11"
    "-DPython3_LIBRARY=$TARGET_PYTHON/lib/libpython3.11.so"
    "-DPython_INCLUDE_DIRS=$TARGET_PYTHON/include/python3.11"
    "-DPython_LIBRARIES=$TARGET_PYTHON/lib/libpython3.11.so"
    "-DPython_NumPy_INCLUDE_DIRS=$TARGET_NUMPY/lib/python3.11/site-packages/numpy/core/include"
    "-DPython3_NumPy_INCLUDE_DIRS=$TARGET_NUMPY/lib/python3.11/site-packages/numpy/core/include"
    "-DPythonModule_mpi4py_PATH=$TARGET_MPI4PY/lib/python3.11/site-packages/mpi4py"
    "-DCMAKE_INSTALL_PYTHONDIR:STRING=lib/python3.11/site-packages"
    "-DSKBUILD_SOABI=$TARGET_SOABI"
    -DADIOS2_USE_PIP=OFF
  )
elif [[ "$PYTHON_MODE" == "on" ]]; then
  ADIOS2_PYTHON_ARGS+=(
    -DADIOS2_USE_Python=ON
    "-DPython_EXECUTABLE=$PYTHON_EXECUTABLE"
  )
else
  ADIOS2_PYTHON_ARGS+=(-DADIOS2_USE_Python=OFF)
fi

cmake -S "$ADIOS2_DIR" -B "$ADIOS2_DIR/build" \
  "${CMAKE_CONFIGURE_ARGS[@]+"${CMAKE_CONFIGURE_ARGS[@]}"}" \
  "${ADIOS2_CROSS_ARGS[@]+"${ADIOS2_CROSS_ARGS[@]}"}" \
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

if [[ "$PYTHON_MODE" == "on" && "$CROSS_PYTHON" != true ]]; then
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

if [[ "$CROSS_PYTHON" == true ]]; then
  CROSS_PYTHON_SITE_PACKAGES=""
  for candidate in "$PREFIX/lib/python3.11/site-packages" "$PREFIX/lib64/python3.11/site-packages"; do
    if [[ -d "$candidate/adios2" ]]; then
      CROSS_PYTHON_SITE_PACKAGES="$candidate"
      break
    fi
  done
  if [[ -z "$CROSS_PYTHON_SITE_PACKAGES" ]]; then
    echo "ADIOS2 Python module was not installed under $PREFIX/lib/python3.11/site-packages" >&2
    exit 1
  fi
  echo "ADIOS2 target Python module: $CROSS_PYTHON_SITE_PACKAGES/adios2"
fi

cat <<EOF

ADIOS2 $ADIOS2_VERSION installed to $PREFIX.

CMake package directory:

  $ADIOS2_CMAKE_DIR
EOF

if [[ "$CROSS_PYTHON" == true ]]; then
  cat <<EOF

Python bindings: cross-compiled for the target Python 3.11.

  $CROSS_PYTHON_SITE_PACKAGES
EOF
elif [[ "$PYTHON_MODE" == "on" ]]; then
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
