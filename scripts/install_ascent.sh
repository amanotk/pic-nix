#!/usr/bin/env bash
set -euo pipefail

ASCENT_VERSION="0.9.5"
CMAKE_MIN_VERSION="3.23"
CMAKE_BOOTSTRAP_VERSION="3.31.10"

usage() {
  cat <<'EOF'
Usage: scripts/install_ascent.sh [install_prefix] --python <python_executable> \
         [--python-is-venv] [--slim|--full]

Build Ascent, Conduit, and Ascent's visualization dependencies with MPI and
Python support. The default installation prefix is "$HOME/usr".

The Python interpreter is required explicitly. By default the installer
creates a virtual environment under <install_prefix>/python-venv and builds
the Conduit and Ascent Python modules for that environment. With
--python-is-venv, the given interpreter must already be a virtual
environment: <install_prefix>/python-venv is linked to that environment so
Conduit and Ascent Python modules install there (no second venv is created).

Profiles:
  --slim   Default when invoked via prepare_build_stack.sh
           Build only what PIC-NIX needs: zlib, Conduit, VTK-m, Ascent.
           Skips HDF5, Silo, ZFP, MFEM, RAJA, Camp, Umpire.
  --full   Upstream build_ascent.sh TPL defaults (all packages above).
           Docs/examples are still disabled (no Sphinx); Cython is required
           in the target environment for ZFP Python bindings.

Examples:

  scripts/install_ascent.sh --python /path/to/python3
  scripts/install_ascent.sh ./thirdparty --python /path/to/python3
  scripts/install_ascent.sh ./thirdparty --python /path/to/venv/bin/python \
    --python-is-venv --slim

Set MPICC and MPICXX to select MPI compiler wrappers. Set
CMAKE_BUILD_PARALLEL_LEVEL to control parallel build jobs (default: 4;
raise it if you have memory headroom).
EOF
}

PREFIX="$HOME/usr"
PYTHON_EXECUTABLE=""
PYTHON_IS_VENV=false
ASCENT_PROFILE="full"

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
  # Devil Ray requires RAJA (which slim does not build). VTK-h still
  # provides scene rendering; keep APComp for compositing.
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
    build_vtkm=true
    build_conduit=true
    build_ascent=true
  )
fi

env "${ASCENT_BUILD_ENV[@]}" "$ASCENT_BUILD_SH"

cat <<EOF

Ascent $ASCENT_VERSION and its dependencies installed to $PREFIX.

Ascent CMake package:

  $PREFIX/ascent-checkout/lib/cmake/ascent

Python environment:

  $PREFIX/python-venv$([[ "$PYTHON_IS_VENV" == true ]] && printf ' -> %s' "$VENV_ROOT")
EOF
