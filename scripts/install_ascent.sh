#!/usr/bin/env bash
set -euo pipefail

ASCENT_VERSION="0.9.5"
CMAKE_MIN_VERSION="3.23"
CMAKE_BOOTSTRAP_VERSION="3.31.10"

usage() {
  cat <<'EOF'
Usage: scripts/install_ascent.sh [install_prefix] --python <python_executable> \
         [--python-is-venv] [--slim|--full] [--rendering]
       scripts/install_ascent.sh [install_prefix] --python <host_python_venv> \
         --cross-python-extracts [--rendering] --cache <fugaku_cache> \
         --target-python <aarch64_python_prefix> --target-numpy <aarch64_numpy_prefix>

Build Ascent, Conduit, and Ascent's visualization dependencies with MPI and
Python support. The default installation prefix is "$HOME/usr".

The Python interpreter is required explicitly. By default the installer
creates a virtual environment under <install_prefix>/python-venv and builds
the Conduit and Ascent Python modules for that environment. With
--python-is-venv, the given interpreter must already be a virtual
environment: <install_prefix>/python-venv is linked to that environment so
Conduit and Ascent Python modules install there (no second venv is created).

Profiles:
  --slim   Default profile (also selected by prepare_build_stack.sh)
           Build only what PIC-NIX needs: zlib, Conduit, and Ascent.
           Skips HDF5, Silo, ZFP, MFEM, RAJA, Camp, Umpire, and rendering.
  --full   Upstream build_ascent.sh TPL defaults (all packages above).
           Docs/examples are still disabled (no Sphinx); Cython is required
           in the target environment for ZFP Python bindings.
  --rendering
           Add VTK-m rendering, VTK-h, and APComp. Supported for native builds
           and Fugaku aarch64 cross builds. Native builds use the selected
           Python virtual environment; cross builds use target Python modules
           and a host build-time environment.
  --cross-python-extracts
           Build Conduit and Ascent with MPI and Python extracts for aarch64.
           The host venv runs configure-time Python; the target prefixes
           provide Python 3.11 and NumPy headers. Add --rendering to also
           build VTK-m and VTK-h for scene rendering and volume rendering.

Examples:

  scripts/install_ascent.sh --python /path/to/python3
  scripts/install_ascent.sh ./thirdparty --python /path/to/python3
  scripts/install_ascent.sh ./thirdparty --python /path/to/venv/bin/python \
     --python-is-venv --slim
  scripts/install_ascent.sh ./thirdparty --python /path/to/python3 \
     --slim --rendering

Set MPICC and MPICXX to select MPI compiler wrappers. Set
CMAKE_BUILD_PARALLEL_LEVEL to control parallel build jobs (default: 4;
raise it if you have memory headroom).
EOF
}

PREFIX="$HOME/usr"
PYTHON_EXECUTABLE=""
PYTHON_IS_VENV=false
ASCENT_PROFILE="slim"
CROSS_PYTHON_EXTRACTS=false
RENDERING=false
CROSS_CACHE=""
TARGET_PYTHON=""
TARGET_NUMPY=""

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
    --python-is-venv)
      PYTHON_IS_VENV=true
      shift
      ;;
    --cross-python-extracts)
      CROSS_PYTHON_EXTRACTS=true
      shift
      ;;
    --rendering)
      RENDERING=true
      shift
      ;;
    --cache|--target-python|--target-numpy)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      case "$1" in
        --cache) CROSS_CACHE="$2" ;;
        --target-python) TARGET_PYTHON="$2" ;;
        --target-numpy) TARGET_NUMPY="$2" ;;
      esac
      shift 2
      ;;
    --slim)
      ASCENT_PROFILE="slim"
      shift
      ;;
    --full)
      ASCENT_PROFILE="full"
      shift
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

if [[ "$CROSS_PYTHON_EXTRACTS" == true ]]; then
  if [[ "$ASCENT_PROFILE" == "full" || "$PYTHON_IS_VENV" != true ]]; then
    echo "--cross-python-extracts requires --slim and --python-is-venv" >&2
    exit 2
  fi
  if [[ -z "$CROSS_CACHE" || -z "$TARGET_PYTHON" || -z "$TARGET_NUMPY" ]]; then
    echo "--cross-python-extracts requires --cache, --target-python, and --target-numpy" >&2
    exit 2
  fi
  for variable in CROSS_CACHE TARGET_PYTHON TARGET_NUMPY; do
    if [[ "${!variable}" != /* ]]; then
      printf -v "$variable" '%s/%s' "$PWD" "${!variable}"
    fi
  done
  if [[ ! -f "$CROSS_CACHE" || ! -f "$TARGET_PYTHON/include/python3.11/Python.h" || ! -f "$TARGET_NUMPY/lib/python3.11/site-packages/numpy/core/include/numpy/arrayobject.h" ]]; then
    echo "Missing cross cache, target Python 3.11 headers, or target NumPy headers" >&2
    exit 2
  fi
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

if [[ "$PYTHON_IS_VENV" != true ]]; then
  if ! "$PYTHON_EXECUTABLE" -m venv --help >/dev/null 2>&1; then
    echo "Python interpreter does not provide the venv module: $PYTHON_EXECUTABLE" >&2
    exit 2
  fi
fi

VENV_ROOT=""
if [[ "$PYTHON_IS_VENV" == true ]]; then
  VENV_ROOT="$("$PYTHON_EXECUTABLE" -c 'import sys; print(sys.prefix)')"
  VENV_BASE="$("$PYTHON_EXECUTABLE" -c 'import sys; print(sys.base_prefix)')"
  if [[ -z "$VENV_ROOT" || "$VENV_ROOT" == "$VENV_BASE" ]]; then
    echo "Python interpreter is not a virtual environment: $PYTHON_EXECUTABLE" >&2
    echo "Omit --python-is-venv to let this script create <install_prefix>/python-venv." >&2
    exit 2
  fi
  if [[ ! -d "$VENV_ROOT" ]]; then
    echo "Virtual environment root not found: $VENV_ROOT" >&2
    exit 2
  fi
fi

# Default 4: VTK-m/icpx at nproc=64 often OOMs on shared nodes.
BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
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

install_cross_python_extracts() {
  local conduit_dir="$BUILDDIR/conduit" ascent_dir="$BUILDDIR/ascent"
  local conduit_prefix="$PREFIX/conduit-v$ASCENT_VERSION"
  local ascent_prefix="$PREFIX/ascent-checkout"
  local python_modules="$PREFIX/python-modules"

  if ! "$PYTHON_EXECUTABLE" -c 'import pip, numpy; assert numpy.__version__ == "1.26.4"; import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
    echo "Cross-build host Python must be 3.11 with pip and NumPy 1.26.4: $PYTHON_EXECUTABLE" >&2
    exit 2
  fi
  if [[ ! -f "$TARGET_PYTHON/lib/libpython3.11.so" ]]; then
    echo "Target Python library not found under $TARGET_PYTHON" >&2
    exit 2
  fi

  git clone https://github.com/LLNL/conduit.git "$conduit_dir" \
    --branch "v$ASCENT_VERSION" --depth 1 --recurse-submodules --shallow-submodules
  git clone https://github.com/Alpine-DAV/ascent.git "$ascent_dir" \
    --branch "v$ASCENT_VERSION" --depth 1 --recurse-submodules --shallow-submodules

  # Upstream derives Python headers and libpython from the executable. Run
  # pure-Python build steps with the x86_64 venv, but compile against the
  # matching aarch64 Python/NumPy ABI. Fail if the pinned upstream layout moves.
  "$PYTHON_EXECUTABLE" - "$conduit_dir" "$ascent_dir" "$TARGET_PYTHON" "$TARGET_NUMPY" <<'PY'
import sys
from pathlib import Path

conduit, ascent, target_python, target_numpy = map(Path, sys.argv[1:])
python_library = target_python / "lib/libpython3.11.so"
python_include = target_python / "include/python3.11"
python_site = target_python / "lib/python3.11/site-packages"
numpy_include = target_numpy / "lib/python3.11/site-packages/numpy/core/include"


def insert_before(path, marker, insertion):
    text = path.read_text()
    if text.count(marker) != 1:
        raise RuntimeError("Unexpected upstream CMake layout: {}".format(path))
    path.write_text(text.replace(marker, insertion + marker, 1))


python_override = (
    "# Use the target Python ABI for cross-compiled extensions.\n"
    'set(PYTHON_LIBRARY "{}")\n'
    'set(PYTHON_INCLUDE_DIR "{}")\n'
    'set(PYTHON_LIBRARY "${{PYTHON_LIBRARY}}" CACHE FILEPATH "" FORCE)\n'
    'set(PYTHON_INCLUDE_DIR "${{PYTHON_INCLUDE_DIR}}" CACHE PATH "" FORCE)\n'
).format(python_library, python_include)
marker = 'MESSAGE(STATUS "{PythonLibs from PythonInterp} using: PYTHON_LIBRARY=${PYTHON_LIBRARY}")'
for project in (conduit, ascent):
    insert_before(
        project / "src/cmake/thirdparty/SetupPython.cmake", marker, python_override
    )

insert_before(
    conduit / "src/cmake/Setup3rdParty.cmake",
    "include(cmake/thirdparty/FindNumPy.cmake)",
    'set(NUMPY_INCLUDE_DIRS "{}")\n'.format(numpy_include),
)
insert_before(
    ascent / "src/cmake/thirdparty/SetupPython.cmake",
    "# for embedded python, we need to know where the site packages dir is",
    'set(PYTHON_SITE_PACKAGES_DIR "{}")\n'.format(python_site),
)
PY

  echo "--- Cross-building Conduit with Python extracts ---"
  cmake -S "$conduit_dir/src" -B "$conduit_dir/build" -C "$CROSS_CACHE" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$conduit_prefix" \
    -DENABLE_MPI=ON -DENABLE_PYTHON=ON -DENABLE_FORTRAN=OFF \
    -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_UTILS=OFF \
    -DENABLE_DOCS=OFF -DENABLE_RELAY_WEBSERVER=ON \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPYTHON_MODULE_INSTALL_PREFIX="$python_modules"
  cmake --build "$conduit_dir/build" --parallel "$BUILD_JOBS"
  cmake --install "$conduit_dir/build"

  local ascent_vtkh=OFF ascent_apcomp=OFF ascent_vtkm_args=()
  if [[ "$RENDERING" == true ]]; then
    local vtkm_dir="$BUILDDIR/vtkm" vtkm_prefix="$PREFIX/vtkm-v2.3.0"
    echo "--- Cross-building VTK-m 2.3.0 with rendering ---"
    git clone https://gitlab.kitware.com/vtk/vtk-m.git "$vtkm_dir" \
      --branch v2.3.0 --depth 1
    git -C "$vtkm_dir" apply \
      "$ascent_dir/scripts/build_ascent/2025_06_18_vtkm_z_extents_ray_culling_bugfix_viskores_mr109.patch"
    cmake -S "$vtkm_dir" -B "$vtkm_dir/build" -C "$CROSS_CACHE" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$vtkm_prefix" \
      -DBUILD_SHARED_LIBS=ON -DVTKm_USE_64BIT_IDS=OFF \
      -DVTKm_USE_DOUBLE_PRECISION=ON \
      -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
      -DVTKm_ENABLE_MPI=ON -DVTKm_ENABLE_OPENMP=ON \
      -DVTKm_ENABLE_RENDERING=ON -DVTKm_ENABLE_TESTING=OFF \
      -DBUILD_TESTING=OFF -DVTKm_ENABLE_BENCHMARKS=OFF
    cmake --build "$vtkm_dir/build" --parallel "$BUILD_JOBS"
    cmake --install "$vtkm_dir/build"
    ascent_vtkh=ON
    ascent_apcomp=ON
    ascent_vtkm_args=(-DVTKM_DIR="$vtkm_prefix")
  fi

  echo "--- Cross-building Ascent with Python extracts ---"
  cmake -S "$ascent_dir/src" -B "$ascent_dir/build" -C "$CROSS_CACHE" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ascent_prefix" \
    -DENABLE_MPI=ON -DENABLE_SERIAL=OFF -DENABLE_PYTHON=ON \
    -DENABLE_FORTRAN=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
    -DENABLE_UTILS=OFF -DENABLE_DOCS=OFF \
    "-DENABLE_VTKH=$ascent_vtkh" \
    "-DENABLE_APCOMP=$ascent_apcomp" \
    -DENABLE_DRAY=OFF \
    "${ascent_vtkm_args[@]+"${ascent_vtkm_args[@]}"}" \
    -DCONDUIT_DIR="$conduit_prefix" \
    -DCONDUIT_PYTHON_MODULE_DIR="$python_modules" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPYTHON_MODULE_INSTALL_PREFIX="$python_modules"
  cmake --build "$ascent_dir/build" --parallel "$BUILD_JOBS"
  cmake --install "$ascent_dir/build"

  [[ -f "$ascent_prefix/lib/cmake/ascent/AscentConfig.cmake" ]] || {
    echo "Ascent CMake package not installed under $ascent_prefix" >&2
    exit 1
  }
  [[ -f "$python_modules/ascent/mpi/ascent_mpi_python.so" ]] || {
    echo "Ascent aarch64 Python module not installed under $python_modules" >&2
    exit 1
  }
  local rendering_desc="without rendering"
  [[ "$RENDERING" == true ]] && rendering_desc="with VTK-h rendering"
  cat <<EOF

Ascent $ASCENT_VERSION (MPI + Python extracts, $rendering_desc) installed to $PREFIX.
Target Python modules: $python_modules
EOF
}

if [[ "$CROSS_PYTHON_EXTRACTS" == true ]]; then
  mkdir -p "$PREFIX"
  install_cross_python_extracts
  exit 0
fi

mkdir -p "$PREFIX"

echo "--- Installing Ascent ($ASCENT_VERSION) ---"
echo "Install prefix: $PREFIX"
echo "Python: $PYTHON_EXECUTABLE"
if [[ "$PYTHON_IS_VENV" == true ]]; then
  echo "Python mode: link $PREFIX/python-venv to existing virtual environment"
else
  echo "Python mode: create $PREFIX/python-venv"
fi
echo "MPI C compiler: $MPICC_EXECUTABLE"
echo "MPI C++ compiler: $MPICXX_EXECUTABLE"
echo "Profile: $ASCENT_PROFILE"
if [[ -n "${CXXFLAGS:-}" ]]; then
  echo "CXXFLAGS: $CXXFLAGS"
fi
if [[ -n "${CXX:-}" ]]; then
  echo "CXX: $CXX"
fi

ASCENT_DIR="$BUILDDIR/ascent"
git clone https://github.com/Alpine-DAV/ascent.git "$ASCENT_DIR" \
  --branch "v$ASCENT_VERSION" --depth 1 --recurse-submodules \
  --shallow-submodules

ASCENT_BUILD_SH="$ASCENT_DIR/scripts/build_ascent/build_ascent.sh"
[[ -f "$ASCENT_BUILD_SH" ]] || {
  echo "build_ascent.sh not found after clone" >&2
  exit 1
}

# PIC-NIX never needs Ascent HTML docs or example apps. Stack venvs do not
# ship sphinx-build; with --python-is-venv the superbuild also skips its
# nested-venv pip bootstrap that would have installed Sphinx. Disable docs
# and examples for every profile (full only changes the TPL set).
echo "--- Disabling Ascent docs/examples (all profiles) ---"
sed -i \
  -e "s/echo 'set(ENABLE_DOCS ON CACHE BOOL \"\")'/echo 'set(ENABLE_DOCS OFF CACHE BOOL \"\")'/" \
  -e "/echo 'set(SPHINX_EXECUTABLE /d" \
  -e "/-DSPHINX_EXECUTABLE=/d" \
  "$ASCENT_BUILD_SH"
if ! grep -q "set(ENABLE_EXAMPLES OFF" "$ASCENT_BUILD_SH"; then
  sed -i \
    "s/echo 'set(ENABLE_TESTS /echo 'set(ENABLE_EXAMPLES OFF CACHE BOOL \"\")'\necho 'set(ENABLE_UTILS OFF CACHE BOOL \"\")'\necho 'set(ENABLE_TESTS /" \
    "$ASCENT_BUILD_SH"
fi
sed -i \
  -e 's|cmake -S ${ascent_src_dir} -B ${ascent_build_dir} -C ${root_dir}/ascent-config.cmake|cmake -S ${ascent_src_dir} -B ${ascent_build_dir} -C ${root_dir}/ascent-config.cmake -DENABLE_EXAMPLES=OFF -DENABLE_UTILS=OFF -DENABLE_TESTS=OFF -DENABLE_DOCS=OFF|' \
  "$ASCENT_BUILD_SH"

if [[ "$ASCENT_PROFILE" == "slim" ]]; then
  echo "--- Applying slim profile patches to build_ascent.sh ---"
  # Unconditional host-config DIR entries for TPLs we do not build.
  sed -i \
    -e "/echo 'set(CAMP_DIR /d" \
    -e "/echo 'set(RAJA_DIR /d" \
    -e "/echo 'set(UMPIRE_DIR /d" \
    -e "/echo 'set(MFEM_DIR /d" \
    "$ASCENT_BUILD_SH"
  # Caliper/Catalyst host-config writes are the sole body of optional
  # if-blocks. Bash rejects empty then-bodies, so remove the whole block.
  # (Only drop blocks that mention the host-config DIR assignment; other
  # if ${build_caliper} blocks, e.g. downloads, must stay.)
  awk '
    /^if \$\{build_caliper\}; then$/ { buf=$0; grab=1; next }
    grab {
      buf = buf "\n" $0
      if ($0 == "fi") {
        if (buf !~ /set\(CALIPER_DIR/) print buf
        grab=0; buf=""
      }
      next
    }
    { print }
    END { if (grab) print buf }
  ' "$ASCENT_BUILD_SH" >"$ASCENT_BUILD_SH.tmp" && cat "$ASCENT_BUILD_SH.tmp" >"$ASCENT_BUILD_SH" && rm -f "$ASCENT_BUILD_SH.tmp"
  awk '
    /^if \$\{build_catalyst\}; then$/ { buf=$0; grab=1; next }
    grab {
      buf = buf "\n" $0
      if ($0 == "fi") {
        if (buf !~ /set\(CATALYST_DIR/) print buf
        grab=0; buf=""
      }
      next
    }
    { print }
    END { if (grab) print buf }
  ' "$ASCENT_BUILD_SH" >"$ASCENT_BUILD_SH.tmp" && cat "$ASCENT_BUILD_SH.tmp" >"$ASCENT_BUILD_SH" && rm -f "$ASCENT_BUILD_SH.tmp"
  # Devil Ray requires RAJA (which slim does not build). Rendering backends
  # are enabled explicitly below instead of being implied by the profile.
  sed -i \
    -e "s/echo 'set(ENABLE_DRAY ON CACHE BOOL \"\")'/echo 'set(ENABLE_DRAY OFF CACHE BOOL \"\")'/" \
    "$ASCENT_BUILD_SH"
  # Conduit is always configured with -DHDF5_DIR even when build_hdf5=false;
  # that path will not exist and FindHDF5 hard-fails. Drop it and keep the
  # Conduit package lean (no examples/utils). Keep ENABLE_RELAY_WEBSERVER:
  # Ascent's web interface requires conduit::relay::web.
  sed -i \
    -e "/-DHDF5_DIR=/d" \
    -e "s/-DENABLE_TESTS=OFF \\\\/-DENABLE_TESTS=OFF \\\\\n  -DENABLE_EXAMPLES=OFF \\\\\n  -DENABLE_UTILS=OFF \\\\/" \
    "$ASCENT_BUILD_SH"
  # Same for Silo (only reached if build_silo=true, but harmless to strip).
  sed -i \
    -e "/-DSILO_ENABLE_HDF5=ON/d" \
    -e "/-DSILO_HDF5_DIR=/d" \
    "$ASCENT_BUILD_SH"
fi

# Upstream enables VTK-m rendering and VTK-h in every native profile. Keep the
# capability independent from the slim/full dependency profile and match the
# cross-build CMake path below. Without rendering, Ascent does not need VTK-m
# or any of the rendering backends.
native_vtkm_rendering=OFF
native_vtkh=OFF
native_apcomp=OFF
native_dray=OFF
if [[ "$RENDERING" == true ]]; then
  native_vtkm_rendering=ON
  native_vtkh=ON
  native_apcomp=ON
fi
sed -E -i \
  -e "s/(VTKm_ENABLE_RENDERING=)[A-Z]+/\\1$native_vtkm_rendering/" \
  -e "s/(set\\(ENABLE_VTKH )[A-Z]+/\\1$native_vtkh/" \
  -e "s/(set\\(ENABLE_APCOMP )[A-Z]+/\\1$native_apcomp/" \
  -e "s/(set\\(ENABLE_DRAY )[A-Z]+/\\1$native_dray/" \
  "$ASCENT_BUILD_SH"
if [[ "$RENDERING" != true ]]; then
  # Do not leave a stale VTK-m prefix in the generated host-config when the
  # non-rendering profile skips the VTK-m build.
  sed -i "/echo 'set(VTKM_DIR /d" "$ASCENT_BUILD_SH"
fi

for feature in \
  "ENABLE_VTKH=$native_vtkh" \
  "ENABLE_APCOMP=$native_apcomp" \
  "ENABLE_DRAY=$native_dray"; do
  if ! grep -Eq "set\\(${feature%%=*} ${feature#*=} CACHE BOOL" "$ASCENT_BUILD_SH"; then
    echo "failed to configure native Ascent feature: $feature" >&2
    exit 1
  fi
done
if ! grep -Eq "VTKm_ENABLE_RENDERING=$native_vtkm_rendering" "$ASCENT_BUILD_SH"; then
  echo "failed to configure native VTK-m rendering: $native_vtkm_rendering" >&2
  exit 1
fi

bash -n "$ASCENT_BUILD_SH" || {
  echo "patch produced invalid build_ascent.sh" >&2
  exit 1
}

# Keep build_pyvenv=true so build_ascent.sh still wires PYTHON_EXECUTABLE and
# Python module install paths. When --python-is-venv is set, pre-create
# <prefix>/python-venv as a symlink to the existing environment: the superbuild
# then skips creating a nested venv (and skips its pip installs) while still
# treating python-venv as the target for Conduit/Ascent Python modules.
BUILD_PYVENV=true
if [[ "$PYTHON_IS_VENV" == true ]]; then
  if [[ -e "$PREFIX/python-venv" && ! -L "$PREFIX/python-venv" ]]; then
    echo "Refusing to replace existing non-symlink: $PREFIX/python-venv" >&2
    echo "Remove it or rerun without --python-is-venv." >&2
    exit 2
  fi
  ln -sfn "$VENV_ROOT" "$PREFIX/python-venv"
  echo "Linked $PREFIX/python-venv -> $VENV_ROOT"
fi

ASCENT_BUILD_ENV=(
  prefix="$BUILDDIR/work"
  install_dir="$PREFIX"
  python_exe="$PYTHON_EXECUTABLE"
  mpicc_exe="$MPICC_EXECUTABLE"
  mpicxx_exe="$MPICXX_EXECUTABLE"
  build_jobs="$BUILD_JOBS"
  build_pyvenv="$BUILD_PYVENV"
  enable_python=ON
  enable_mpi=ON
  enable_mpicc=ON
  enable_openmp=ON
  enable_fortran=OFF
  enable_tests=OFF
)

if [[ "$ASCENT_PROFILE" == "slim" ]]; then
  ASCENT_BUILD_ENV+=(
    build_zlib=true
    build_hdf5=false
    build_silo=false
    build_zfp=false
    build_mfem=false
    build_raja=false
    build_umpire=false
    build_camp=false
    build_caliper=false
    build_catalyst=false
    build_vtkm=false
    build_conduit=true
    build_ascent=true
  )
fi

if [[ "$RENDERING" == true ]]; then
  ASCENT_BUILD_ENV+=(build_vtkm=true)
else
  ASCENT_BUILD_ENV+=(build_vtkm=false)
fi

env "${ASCENT_BUILD_ENV[@]}" "$ASCENT_BUILD_SH"

cat <<EOF

Ascent $ASCENT_VERSION (MPI + Python extracts, \
$([[ "$RENDERING" == true ]] && printf 'with' || printf 'without') VTK-h rendering) installed to $PREFIX.

Ascent CMake package:

  $PREFIX/ascent-checkout/lib/cmake/ascent

Python environment:

  $PREFIX/python-venv$([[ "$PYTHON_IS_VENV" == true ]] && printf ' -> %s' "$VENV_ROOT")
EOF
