#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STACK_SCRIPT_VERSION="1"

usage() {
  cat <<'EOF'
Usage: scripts/prepare_build_stack.sh <stack_dir> [options]

Prepare a reusable PIC-NIX build stack in <stack_dir> (any path, typically
outside the repository). The directory must not be moved after creation:
stamps, env.sh, and CMake caches store absolute paths.

By default this installs:
  - a uv-managed virtual environment (system Python preferred) with
    mpi4py, numpy, and an editable picnix install (cross-builds omit mpi4py)
  - the ordinary C++ dependencies into <stack_dir>/deps

When --with-ascent is used, the Ascent superbuild runs in **slim** profile
(zlib + Conduit + VTK-m + Ascent only; no HDF5/Silo/ZFP/MFEM/RAJA/Sphinx).
Use --ascent-full for upstream's full third-party set (still no Sphinx/docs;
Cython only). 

Optional long builds (off by default):
  --with-adios2         ADIOS2 C++/MPI library (Python bindings off)
  --with-adios2-python  ADIOS2 Python bindings as well (implies --with-adios2)
  --with-ascent         Ascent + Conduit (slim); Python modules go into the stack venv

Options:
  --cache <file.cmake>     CMake initial-cache that selects the compiler
                           (required unless both --mpicc and --mpicxx are given)
  --mpicc <path>           Override the MPI C wrapper
  --mpicxx <path>          Override the MPI C++ wrapper
  --python-version <X.Y>   Fallback interpreter when no suitable system Python
                           exists (default: 3.12)
  --with-adios2            Build ADIOS2 (C++ only)
  --with-adios2-python     Build ADIOS2 with Python bindings
  --with-ascent            Build Ascent into the stack Python environment
  --ascent-extracts-only   With --with-ascent and a Fugaku aarch64 cache:
                            cross-build MPI + Python extracts without rendering
  --ascent-full            With --with-ascent: upstream full TPL set (slow;
                           no Sphinx/docs; needs Cython for ZFP)
  --no-deps                Skip the ordinary C++ dependencies
  --no-picnix              Skip the editable picnix install
  --check                  Validate an existing stack; do not build
  --force                  Ignore stamps and rebuild components
  --jobs <N>               Parallel build jobs (default: 4; env
                           CMAKE_BUILD_PARALLEL_LEVEL overrides)
  -h, --help               Show this help

After a successful run:

  source <stack_dir>/env.sh

The same compiler fingerprint must be used when configuring PIC-NIX itself.
Example (default stack):

  cmake -S . -B build -C cmake/linux-gcc.cmake \
    -DCMAKE_PREFIX_PATH=<stack_dir>/deps \
    -DPICNIX_USE_SYSTEM_LIBS=ON

Cross-compilation caches may build the default stack. Fugaku aarch64 caches
also support --with-adios2 (C++/MPI only) and --ascent-extracts-only (MPI and
Python extracts, without rendering). --with-adios2-python and Ascent's
rendering/full profiles require a native build.

Parallelism defaults to 4 jobs to avoid OOM on large hosts. Raise it with
--jobs N or CMAKE_BUILD_PARALLEL_LEVEL if you have memory headroom.
EOF
}

STACK_DIR=""
CACHE_FILE=""
MPICC_EXPLICIT=""
MPICXX_EXPLICIT=""
PYTHON_FALLBACK_VERSION="3.12"
WITH_ADIOS2=false
WITH_ADIOS2_PYTHON=false
WITH_ASCENT=false
ASCENT_FULL=false
ASCENT_EXTRACTS_ONLY=false
WITH_DEPS=true
WITH_PICNIX=true
CHECK_ONLY=false
FORCE=false
# Conservative default: large nproc values thrash memory on icpx/VTK-m
# builds (OOM / compiler killed). Override with --jobs or
# CMAKE_BUILD_PARALLEL_LEVEL.
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

while (( $# > 0 )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --cache)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      CACHE_FILE="$2"
      shift 2
      ;;
    --cache=*)
      CACHE_FILE="${1#*=}"
      shift
      ;;
    --mpicc)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      MPICC_EXPLICIT="$2"
      shift 2
      ;;
    --mpicc=*)
      MPICC_EXPLICIT="${1#*=}"
      shift
      ;;
    --mpicxx)
      if (( $# < 2 )); then
        echo "Missing path after $1" >&2
        exit 2
      fi
      MPICXX_EXPLICIT="$2"
      shift 2
      ;;
    --mpicxx=*)
      MPICXX_EXPLICIT="${1#*=}"
      shift
      ;;
    --python-version)
      if (( $# < 2 )); then
        echo "Missing version after $1" >&2
        exit 2
      fi
      PYTHON_FALLBACK_VERSION="$2"
      shift 2
      ;;
    --python-version=*)
      PYTHON_FALLBACK_VERSION="${1#*=}"
      shift
      ;;
    --with-adios2)
      WITH_ADIOS2=true
      shift
      ;;
    --with-adios2-python)
      WITH_ADIOS2=true
      WITH_ADIOS2_PYTHON=true
      shift
      ;;
    --with-ascent)
      WITH_ASCENT=true
      shift
      ;;
    --ascent-extracts-only)
      WITH_ASCENT=true
      ASCENT_EXTRACTS_ONLY=true
      shift
      ;;
    --ascent-full)
      WITH_ASCENT=true
      ASCENT_FULL=true
      shift
      ;;
    --no-deps)
      WITH_DEPS=false
      shift
      ;;
    --no-picnix)
      WITH_PICNIX=false
      shift
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --jobs)
      if (( $# < 2 )); then
        echo "Missing count after $1" >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$STACK_DIR" ]]; then
        STACK_DIR="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$STACK_DIR" ]]; then
  echo "stack_dir is required" >&2
  usage >&2
  exit 2
fi

if [[ "$STACK_DIR" != /* ]]; then
  STACK_DIR="$PWD/$STACK_DIR"
fi

if [[ "$CHECK_ONLY" != true && -z "$CACHE_FILE" && ( -z "$MPICC_EXPLICIT" || -z "$MPICXX_EXPLICIT" ) ]]; then
  echo "--cache <file.cmake> is required unless both --mpicc and --mpicxx are given" >&2
  usage >&2
  exit 2
fi

if [[ -n "$CACHE_FILE" && "$CACHE_FILE" != /* ]]; then
  CACHE_FILE="$PWD/$CACHE_FILE"
fi

if [[ -n "$CACHE_FILE" && ! -f "$CACHE_FILE" ]]; then
  echo "Cache file not found: $CACHE_FILE" >&2
  exit 2
fi

if [[ -z "$JOBS" ]]; then
  JOBS=4
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

STACK_PYTHON="$STACK_DIR/python"
STACK_VENV_BIN="$STACK_PYTHON/bin/python"
STACK_DEPS="$STACK_DIR/deps"
STACK_ADIOS2="$STACK_DIR/adios2"
STACK_ASCENT="$STACK_DIR/ascent"
STAMP_DIR="$STACK_DIR/stamp"
ENV_SH="$STACK_DIR/env.sh"
ENV_LOCAL_SH="$STACK_DIR/env.local.sh"
LOCK_FILE="$STACK_DIR/.lock"
SITE_PACKAGES=""

log() { printf '%s\n' "$*"; }
err() { printf '%s\n' "$*" >&2; }
die() { err "error: $*"; exit 1; }

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "required command not found: sha256sum or shasum"
  fi
}

hash_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
  else
    die "required command not found: sha256sum or shasum"
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "required command not found: $1"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    return 0
  fi
  log "--- Bootstrapping uv into ~/.local ---"
  require_command curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  require_command uv
}

cache_cxx_compiler() {
  local value=""
  value="$(awk -F'"' '
    /^[[:space:]]*set[[:space:]]*\([[:space:]]*CMAKE_CXX_COMPILER[[:space:]]+"/ {
      print $2
      exit
    }
  ' "$CACHE_FILE")"
  if [[ -z "$value" ]]; then
    value="$(awk '
      /^[[:space:]]*set[[:space:]]*\([[:space:]]*CMAKE_CXX_COMPILER[[:space:]]+/ {
        line = $0
        sub(/^[[:space:]]*set[[:space:]]*\([[:space:]]*CMAKE_CXX_COMPILER[[:space:]]+/, "", line)
        sub(/[).].*$/, "", line)
        gsub(/^"|"$/, "", line)
        print line
        exit
      }
    ' "$CACHE_FILE")"
  fi
  printf '%s' "$value"
}

cache_system_name() {
  awk -F'"' '
    /^[[:space:]]*set[[:space:]]*\([[:space:]]*CMAKE_SYSTEM_NAME[[:space:]]+"/ {
      print $2
      exit
    }
  ' "$CACHE_FILE"
}

cache_system_processor() {
  awk -F'"' '
    /^[[:space:]]*set[[:space:]]*\([[:space:]]*CMAKE_SYSTEM_PROCESSOR[[:space:]]+"/ {
      print $2
      exit
    }
  ' "$CACHE_FILE"
}

# Read a quoted `set(KEY "VALUE" ...)` entry from the initial-cache file.
cache_quoted_var() {
  local key="$1"
  if [[ -z "$CACHE_FILE" || ! -f "$CACHE_FILE" ]]; then
    return 1
  fi
  awk -F'"' -v key="$key" '
    $0 ~ ("set[[:space:]]*\\([[:space:]]*" key "[[:space:]]+") && index($0, "\"") {
      print $2
      exit
    }
  ' "$CACHE_FILE"
}

# Export CC/CXX/CFLAGS/CXXFLAGS from the initial-cache so child builds that
# cannot take -C (Ascent superbuild) still use the same compiler flags.
# Critical for cmake/linux-intel.cmake, where mpiicpc only selects icpx via
# -cxx=icpx in CMAKE_CXX_FLAGS.
apply_cache_compiler_env() {
  local cxxflags cflags cxx_compiler c_compiler
  if [[ -z "$CACHE_FILE" ]]; then
    return 0
  fi
  cxxflags="$(cache_quoted_var CMAKE_CXX_FLAGS || true)"
  cflags="$(cache_quoted_var CMAKE_C_FLAGS || true)"
  cxx_compiler="$(cache_cxx_compiler || true)"
  c_compiler=""
  if [[ -n "$cxx_compiler" ]]; then
    c_compiler="$(derive_mpicc_from_mpicxx "$cxx_compiler")"
  fi
  if [[ -n "$cxxflags" ]]; then
    export CXXFLAGS="$cxxflags"
  fi
  if [[ -n "$cflags" ]]; then
    export CFLAGS="$cflags"
  fi
  if [[ -n "$cxx_compiler" ]]; then
    if [[ "$cxx_compiler" != */* ]]; then
      cxx_compiler="$(command -v "$cxx_compiler" || true)"
    elif [[ "$cxx_compiler" != /* ]]; then
      cxx_compiler="$(command -v "$cxx_compiler" || printf '%s' "$cxx_compiler")"
      if [[ "$cxx_compiler" != /* && "$cxx_compiler" == ./* ]]; then
        cxx_compiler="$(command -v "${cxx_compiler#./}" || true)"
      fi
    fi
    if [[ -n "$cxx_compiler" ]]; then
      export CXX="$cxx_compiler"
    fi
  fi
  if [[ -n "$c_compiler" ]]; then
    if [[ "$c_compiler" != */* ]]; then
      c_compiler="$(command -v "$c_compiler" || true)"
    else
      c_compiler="$(command -v "$c_compiler" || command -v "${c_compiler#./}" || true)"
    fi
    if [[ -n "$c_compiler" ]]; then
      export CC="$c_compiler"
    fi
  fi
}

derive_mpicc_from_mpicxx() {
  local mpicxx="$1"
  local dir base
  if [[ "$mpicxx" == */* ]]; then
    dir="$(dirname "$mpicxx")"
    base="$(basename "$mpicxx")"
  else
    dir=""
    base="$mpicxx"
  fi
  case "$base" in
    mpicxx|mpic++)
      if [[ -n "$dir" ]]; then printf '%s/%s' "$dir" mpicc; else printf '%s' mpicc; fi
      ;;
    mpiicpc)
      if [[ -n "$dir" ]]; then printf '%s/%s' "$dir" mpiicc; else printf '%s' mpiicc; fi
      ;;
    mpiclang++)
      if [[ -n "$dir" ]]; then printf '%s/%s' "$dir" mpiclang; else printf '%s' mpiclang; fi
      ;;
    mpicxx.*)
      if [[ -n "$dir" ]]; then
        printf '%s/%s' "$dir" "mpicc.${base#mpicxx.}"
      else
        printf '%s' "mpicc.${base#mpicxx.}"
      fi
      ;;
    *)
      if [[ -n "$dir" ]]; then printf '%s/%s' "$dir" mpicc; else printf '%s' mpicc; fi
      ;;
  esac
}

underlying_compiler_id() {
  local wrapper="$1"
  local shown=""
  if shown="$("$wrapper" --showme:command 2>/dev/null)" && [[ -n "$shown" ]]; then
    printf '%s' "$shown"
    return 0
  fi
  if shown="$("$wrapper" -show 2>/dev/null)" && [[ -n "$shown" ]]; then
    printf '%s' "${shown%% *}"
    return 0
  fi
  printf '%s' "$wrapper"
}

resolve_compiler() {
  local cxx_wrapper=""
  if [[ -n "$MPICXX_EXPLICIT" ]]; then
    cxx_wrapper="$MPICXX_EXPLICIT"
  elif [[ -n "$CACHE_FILE" ]]; then
    cxx_wrapper="$(cache_cxx_compiler)"
  fi
  if [[ -z "$cxx_wrapper" ]]; then
    die "could not resolve the C++ MPI wrapper; pass --mpicxx"
  fi
  local requested="$cxx_wrapper"
  if [[ "$cxx_wrapper" != */* ]]; then
    cxx_wrapper="$(command -v "$cxx_wrapper" || true)"
  fi
  if [[ -z "$cxx_wrapper" || ! -x "$cxx_wrapper" ]]; then
    die "MPI C++ wrapper not found: $requested (from --mpicxx or cache); load the matching MPI module or pass --mpicxx"
  fi

  local c_wrapper=""
  if [[ -n "$MPICC_EXPLICIT" ]]; then
    c_wrapper="$MPICC_EXPLICIT"
  else
    c_wrapper="$(derive_mpicc_from_mpicxx "$cxx_wrapper")"
  fi
  if [[ "$c_wrapper" != */* ]]; then
    c_wrapper="$(command -v "$c_wrapper" || true)"
  fi
  [[ -n "$c_wrapper" && -x "$c_wrapper" ]] || die "MPI C wrapper not found (derived from $cxx_wrapper); pass --mpicc"

  MPICC_EXECUTABLE="$c_wrapper"
  MPICXX_EXECUTABLE="$cxx_wrapper"
}

build_fingerprint() {
  local underlying version_line cache_hash system_name system_proc
  underlying="$(underlying_compiler_id "$MPICXX_EXECUTABLE")"
  version_line="$("$MPICXX_EXECUTABLE" --version 2>/dev/null | head -n 1 || true)"
  cache_hash=""
  system_name=""
  system_proc=""
  if [[ -n "$CACHE_FILE" ]]; then
    cache_hash="$(hash_file "$CACHE_FILE")"
    system_name="$(cache_system_name)"
    system_proc="$(cache_system_processor)"
  fi
  HOST_UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
  HOST_UNAME_M="$(uname -m 2>/dev/null || echo unknown)"
  IS_CROSS=false
  # A named CMAKE_SYSTEM_NAME other than the host OS (or a differing
  # CMAKE_SYSTEM_PROCESSOR) indicates a cross/initial-cache target build.
  if [[ -n "$system_name" && "$system_name" != "$HOST_UNAME_S" ]]; then
    IS_CROSS=true
  fi
  if [[ -n "$system_proc" && "$system_proc" != "$HOST_UNAME_M" ]]; then
    IS_CROSS=true
  fi

  COMPILER_FINGERPRINT="$(hash_text "mpicc=$MPICC_EXECUTABLE
mpicxx=$MPICXX_EXECUTABLE
underlying=$underlying
version=$version_line
cache=$CACHE_FILE
cache_hash=$cache_hash
system_name=$system_name
system_proc=$system_proc
host=$HOST_UNAME_S-$HOST_UNAME_M
stack_script=$STACK_SCRIPT_VERSION")"
}

stamp_path() { printf '%s/%s.stamp' "$STAMP_DIR" "$1"; }

stamp_matches() {
  local name="$1" expected="$2"
  local path
  path="$(stamp_path "$name")"
  [[ -f "$path" ]] || return 1
  [[ "$(cat "$path")" == "$expected" ]]
}

write_stamp() {
  local name="$1" value="$2"
  mkdir -p "$STAMP_DIR"
  printf '%s\n' "$value" >"$(stamp_path "$name")"
}

clear_stamp() {
  rm -f "$(stamp_path "$1")"
}

compute_stamp() {
  local name="$1"
  shift
  local payload="component=$name
stack_script=$STACK_SCRIPT_VERSION
fingerprint=$COMPILER_FINGERPRINT
python_fallback=$PYTHON_FALLBACK_VERSION
jobs_ignored=1"
  local item
  for item in "$@"; do
    case "$item" in
      script:*)
        payload+=$'\n'"script_hash=$(hash_file "${item#script:}")"
        ;;
      cache:*)
        if [[ -n "${item#cache:}" ]]; then
          payload+=$'\n'"cache_hash=$(hash_file "${item#cache:}")"
        fi
        ;;
      *)
        payload+=$'\n'"$item"
        ;;
    esac
  done
  hash_text "$payload"
}

# Plain-text sidecar for flags that --check cannot recompute from the hash alone.
write_stamp_meta() {
  local name="$1"
  shift
  mkdir -p "$STAMP_DIR"
  printf '%s\n' "$@" >"$STAMP_DIR/${name}.meta"
}

venv_python_tag() {
  if [[ -x "$STACK_VENV_BIN" ]]; then
    "$STACK_VENV_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")' 2>/dev/null || true
  fi
}

ensure_lock() {
  mkdir -p "$STACK_DIR"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    die "another prepare_build_stack.sh is using $STACK_DIR (lock: $LOCK_FILE)"
  fi
}

select_system_python() {
  local candidate min_major min_minor major minor
  candidate=""
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    [[ -n "$candidate" ]] || continue
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    if ! "$candidate" -c 'import sys' >/dev/null 2>&1; then
      continue
    fi
    read -r major minor < <("$candidate" -c 'import sys; print(sys.version_info[0], sys.version_info[1])')
    min_major=3
    min_minor=10
    if (( major > min_major || (major == min_major && minor >= min_minor) )); then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

prepare_python() {
  ensure_uv
  local desired="" py=""
  mapfile -t PYTHON_CANDIDATES < <(
    if [[ -n "${PYTHON_EXPLICIT:-}" ]]; then
      printf '%s\n' "$PYTHON_EXPLICIT"
    fi
    printf '%s\n' "python3" "python"
  )
  if py="$(select_system_python)"; then
    desired="$py"
    log "--- Python: system interpreter $desired ---"
  else
    desired="$PYTHON_FALLBACK_VERSION"
    log "--- Python: no suitable system interpreter; using uv-managed $desired ---"
  fi

  local py_packages="mpi4py,numpy,pip,setuptools,wheel"
  if [[ "$IS_CROSS" == true ]]; then
    # The stack Python runs on the login node; cross-built mpi4py cannot be
    # imported there. Keep the target MPI wrappers for C++ dependencies only.
    py_packages="numpy,pip,setuptools,wheel"
  fi
  if [[ "$WITH_ASCENT" == true && "$ASCENT_FULL" == true ]]; then
    # Full Ascent profile builds ZFP Python bindings and needs Cython.
    # Slim profile (default) skips ZFP entirely.
    py_packages+=",cython"
  fi
  if [[ "$WITH_PICNIX" == true ]]; then
    py_packages+=",picnix"
  fi

  local expected
  local stamp_inputs=(
    "with_picnix=$WITH_PICNIX"
    "with_adios2=$WITH_ADIOS2"
    "with_adios2_python=$WITH_ADIOS2_PYTHON"
    "with_ascent=$WITH_ASCENT"
    "ascent_full=$ASCENT_FULL"
    "desired_python=$desired"
    "packages=$py_packages"
  )
  if [[ "$WITH_PICNIX" == true && -f "$REPO_ROOT/python/pyproject.toml" ]]; then
    # Editable installs do not pick up new declared deps/entry points.
    stamp_inputs+=("pyproject_hash=$(hash_file "$REPO_ROOT/python/pyproject.toml")")
  fi
  expected="$(compute_stamp python "${stamp_inputs[@]}")"

  if [[ "$FORCE" == true ]]; then
    clear_stamp python
  fi

  # Soft-refresh: reinstall packages into an existing venv when only stamp
  # inputs changed (pyproject, extras). Full recreate only when the
  # interpreter is missing or FORCE — wiping the venv would delete Conduit/
  # Ascent extensions that Ascent installed into it.
  local need_recreate=true
  if [[ -x "$STACK_VENV_BIN" && "$FORCE" != true ]]; then
    local cur_desired
    cur_desired="$("$STACK_VENV_BIN" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    # Same venv still points at a working interpreter: soft path.
    if [[ -n "$cur_desired" ]]; then
      need_recreate=false
    fi
  fi

  if [[ "$need_recreate" == false ]] && stamp_matches python "$expected"; then
    log "Python environment up to date: $STACK_VENV_BIN"
    write_stamp_meta python \
      "with_picnix=$WITH_PICNIX" \
      "with_adios2=$WITH_ADIOS2" \
      "with_adios2_python=$WITH_ADIOS2_PYTHON" \
      "with_ascent=$WITH_ASCENT" \
      "ascent_full=$ASCENT_FULL" \
      "desired_python=$desired" \
      "packages=$py_packages"
    if [[ "$WITH_PICNIX" == true && -f "$REPO_ROOT/python/pyproject.toml" ]]; then
      printf 'pyproject_hash=%s\n' "$(hash_file "$REPO_ROOT/python/pyproject.toml")" >>"$STAMP_DIR/python.meta"
    fi
  elif [[ "$need_recreate" == false ]]; then
    log "--- Refreshing Python packages in existing venv ---"
    local base_pkgs=(numpy setuptools wheel)
    if [[ "$IS_CROSS" != true ]]; then
      base_pkgs=(mpi4py "${base_pkgs[@]}")
    fi
    if [[ "$WITH_ASCENT" == true && "$ASCENT_FULL" == true ]]; then
      base_pkgs+=(cython)
    fi
    if [[ "$IS_CROSS" != true && -n "$MPICC_EXECUTABLE" ]]; then
      MPICC="$MPICC_EXECUTABLE" uv pip install --python "$STACK_VENV_BIN" --no-binary mpi4py "${base_pkgs[@]}"
    else
      uv pip install --python "$STACK_VENV_BIN" --no-binary mpi4py "${base_pkgs[@]}"
    fi
    if [[ "$WITH_PICNIX" == true ]]; then
      local extras="mpi,test"
      if [[ "$IS_CROSS" == true ]]; then
        extras="test"
      fi
      if [[ "$WITH_ADIOS2" == true && "$WITH_ADIOS2_PYTHON" != true ]]; then
        extras="${extras},adios"
      fi
      uv pip install --python "$STACK_VENV_BIN" -e "$REPO_ROOT/python[$extras]"
    fi
    write_stamp python "$expected"
    write_stamp_meta python \
      "with_picnix=$WITH_PICNIX" \
      "with_adios2=$WITH_ADIOS2" \
      "with_adios2_python=$WITH_ADIOS2_PYTHON" \
      "with_ascent=$WITH_ASCENT" \
      "ascent_full=$ASCENT_FULL" \
      "desired_python=$desired" \
      "packages=$py_packages"
    if [[ "$WITH_PICNIX" == true && -f "$REPO_ROOT/python/pyproject.toml" ]]; then
      printf 'pyproject_hash=%s\n' "$(hash_file "$REPO_ROOT/python/pyproject.toml")" >>"$STAMP_DIR/python.meta"
    fi
  else
    log "--- Creating virtual environment at $STACK_PYTHON ---"
    rm -rf "$STACK_PYTHON"
    # Wiping the venv deletes Conduit/Ascent Python modules Ascent installed
    # into it; force those components to rebuild against the new interpreter.
    clear_stamp ascent
    if [[ "$WITH_ADIOS2_PYTHON" == true ]]; then
      clear_stamp adios2
    fi
    # --seed adds pip; Conduit's superbuild runs
    # `python -m pip install . --no-build-isolation`, which also needs
    # setuptools and wheel already present in the environment.
    uv venv "$STACK_PYTHON" --python "$desired" --seed
    [[ -x "$STACK_VENV_BIN" ]] || die "failed to create virtual environment at $STACK_PYTHON"

    local base_pkgs=(numpy setuptools wheel)
    if [[ "$IS_CROSS" != true ]]; then
      base_pkgs=(mpi4py "${base_pkgs[@]}")
    fi
    if [[ "$WITH_ASCENT" == true && "$ASCENT_FULL" == true ]]; then
      base_pkgs+=(cython)
    fi
    log "--- Installing Python packages (${base_pkgs[*]}) ---"
    if [[ "$IS_CROSS" != true && -n "$MPICC_EXECUTABLE" ]]; then
      MPICC="$MPICC_EXECUTABLE" uv pip install --python "$STACK_VENV_BIN" --no-binary mpi4py "${base_pkgs[@]}"
    else
      uv pip install --python "$STACK_VENV_BIN" --no-binary mpi4py "${base_pkgs[@]}"
    fi

    if [[ "$WITH_PICNIX" == true ]]; then
      local extras="mpi,test"
      if [[ "$IS_CROSS" == true ]]; then
        extras="test"
      fi
      if [[ "$WITH_ADIOS2" == true && "$WITH_ADIOS2_PYTHON" != true ]]; then
        extras="${extras},adios"
      fi
      log "--- Installing editable picnix [$extras] ---"
      uv pip install --python "$STACK_VENV_BIN" -e "$REPO_ROOT/python[$extras]"
      if [[ "$WITH_ADIOS2_PYTHON" == true ]]; then
        uv pip uninstall --python "$STACK_VENV_BIN" adios2 >/dev/null 2>&1 || true
      fi
    fi

    write_stamp python "$expected"
    write_stamp_meta python \
      "with_picnix=$WITH_PICNIX" \
      "with_adios2=$WITH_ADIOS2" \
      "with_adios2_python=$WITH_ADIOS2_PYTHON" \
      "with_ascent=$WITH_ASCENT" \
      "ascent_full=$ASCENT_FULL" \
      "desired_python=$desired" \
      "packages=$py_packages"
    if [[ "$WITH_PICNIX" == true && -f "$REPO_ROOT/python/pyproject.toml" ]]; then
      printf 'pyproject_hash=%s\n' "$(hash_file "$REPO_ROOT/python/pyproject.toml")" >>"$STAMP_DIR/python.meta"
    fi
  fi

  SITE_PACKAGES="$("$STACK_VENV_BIN" -c 'import site; print(site.getsitepackages()[0])')"
}

install_deps_component() {
  if [[ "$WITH_DEPS" != true ]]; then
    return 0
  fi
  local expected
  local stamp_items=(
    "script:$REPO_ROOT/scripts/install_dependencies.sh"
  )
  if [[ -n "$CACHE_FILE" ]]; then
    stamp_items+=("cache:$CACHE_FILE")
  fi
  expected="$(compute_stamp deps "${stamp_items[@]}")"
  if [[ "$FORCE" == true ]]; then
    clear_stamp deps
  fi
  if [[ -d "$STACK_DEPS" ]] && stamp_matches deps "$expected"; then
    log "C++ dependencies up to date: $STACK_DEPS"
    write_stamp_meta deps "${stamp_items[@]}"
    return 0
  fi
  log "--- Installing C++ dependencies into $STACK_DEPS ---"
  local args=("$STACK_DEPS")
  if [[ -n "$CACHE_FILE" ]]; then
    args+=(-C "$CACHE_FILE")
  fi
  "$REPO_ROOT/scripts/install_dependencies.sh" "${args[@]}"
  write_stamp deps "$expected"
  write_stamp_meta deps "${stamp_items[@]}"
}

install_adios2_component() {
  if [[ "$WITH_ADIOS2" != true ]]; then
    return 0
  fi
  local py_tag=""
  if [[ "$WITH_ADIOS2_PYTHON" == true ]]; then
    py_tag="$(venv_python_tag)"
  fi
  local expected
  expected="$(compute_stamp adios2 \
    "script:$REPO_ROOT/scripts/install_adios2.sh" \
    "python_mode=$([[ "$WITH_ADIOS2_PYTHON" == true ]] && echo on || echo off)")"
  if [[ "$FORCE" == true ]]; then
    clear_stamp adios2
  fi
  # ABI check: native bindings must match the current interpreter even if
  # other stamp inputs are unchanged.
  if [[ "$WITH_ADIOS2_PYTHON" == true && -f "$STAMP_DIR/adios2.meta" && -n "$py_tag" ]]; then
    local old_py
    old_py="$(awk -F= '/^venv_python=/ {print $2}' "$STAMP_DIR/adios2.meta" 2>/dev/null || true)"
    if [[ -n "$old_py" && "$old_py" != "$py_tag" ]]; then
      log "ADIOS2 Python ABI changed ($old_py -> $py_tag); invalidating stamp"
      clear_stamp adios2
    fi
  fi
  if [[ -d "$STACK_ADIOS2" ]] && stamp_matches adios2 "$expected"; then
    log "ADIOS2 up to date: $STACK_ADIOS2"
    write_stamp_meta adios2 \
      "python_mode=$([[ "$WITH_ADIOS2_PYTHON" == true ]] && echo on || echo off)" \
      "venv_python=$py_tag"
    return 0
  fi
  log "--- Installing ADIOS2 into $STACK_ADIOS2 ---"
  local args=("$STACK_ADIOS2")
  if [[ "$WITH_ADIOS2_PYTHON" == true ]]; then
    args+=(--python "$STACK_VENV_BIN")
  else
    args+=(--no-python)
  fi
  if [[ -n "$CACHE_FILE" ]]; then
    args+=(-C "$CACHE_FILE")
  fi
  if [[ "$IS_CROSS" == true ]]; then
    args+=(
      --cross-build
      -DFFS_FLOAT_FORMAT_TEST:STRING=0
      -DFFS_FLOAT_FORMAT_TEST__TRYRUN_OUTPUT:STRING=Format_IEEE_754_littleendian
      -DADIOS2_USE_MHS=OFF -DADIOS2_USE_PNG=OFF
      -DADIOS2_USE_Sodium=OFF -DADIOS2_USE_OpenSSL=OFF
      -DADIOS2_USE_CURL=OFF -DADIOS2_USE_Campaign=OFF
      -DADIOS2_USE_Profiling=OFF
    )
  fi
  apply_cache_compiler_env
  MPICC="$MPICC_EXECUTABLE" MPICXX="$MPICXX_EXECUTABLE" \
    "$REPO_ROOT/scripts/install_adios2.sh" "${args[@]}"
  write_stamp adios2 "$expected"
  write_stamp_meta adios2 \
    "python_mode=$([[ "$WITH_ADIOS2_PYTHON" == true ]] && echo on || echo off)" \
    "venv_python=$py_tag"
}

find_adios2_site_packages() {
  local libdir candidate version
  for libdir in "$STACK_ADIOS2/lib" "$STACK_ADIOS2/lib64"; do
    [[ -d "$libdir" ]] || continue
    for candidate in "$libdir"/python*/site-packages; do
      if [[ -d "$candidate/adios2" ]]; then
        printf '%s' "$candidate"
        return 0
      fi
    done
    version="$("$STACK_VENV_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
    if [[ -n "$version" && -d "$libdir/python${version}/site-packages/adios2" ]]; then
      printf '%s' "$libdir/python${version}/site-packages"
      return 0
    fi
  done
  return 1
}

wire_python_paths() {
  [[ -n "$SITE_PACKAGES" && -d "$SITE_PACKAGES" ]] || return 0
  local pth="$SITE_PACKAGES/zz-picnix-stack.pth"
  local lines=()
  if [[ "$WITH_ADIOS2" == true && "$WITH_ADIOS2_PYTHON" == true ]]; then
    local sp
    if sp="$(find_adios2_site_packages)"; then
      lines+=("$sp")
    else
      die "ADIOS2 Python site-packages not found under $STACK_ADIOS2"
    fi
  fi
  if [[ "$WITH_ASCENT" == true && "$ASCENT_EXTRACTS_ONLY" != true ]]; then
    if ! "$STACK_VENV_BIN" -c 'import conduit' >/dev/null 2>&1; then
      local candidate
      for candidate in \
        "$STACK_ASCENT"/python-venv/lib/python*/site-packages \
        "$STACK_ASCENT"/ascent-checkout/lib/python*/site-packages
      do
        if [[ -d "$candidate/conduit" || -d "$candidate/ascent" ]]; then
          lines+=("$candidate")
          break
        fi
      done
    fi
    if [[ "${#lines[@]}" -eq 0 ]] && ! "$STACK_VENV_BIN" -c 'import conduit' >/dev/null 2>&1; then
      err "warning: Conduit/Ascent Python modules not importable from the stack venv;"
      err "         check $STACK_ASCENT layout and extend $pth manually if needed"
    fi
  fi
  if [[ "${#lines[@]}" -gt 0 ]]; then
    printf '%s\n' "${lines[@]}" >"$pth"
    log "Wrote $pth"
  elif [[ -f "$pth" && "$WITH_ADIOS2_PYTHON" != true ]]; then
    rm -f "$pth"
  fi
}

spack_public_prefix() {
  local setup="/vol0004/apps/oss/spack/share/spack/setup-env.sh"
  [[ -f "$setup" ]] || die "Fugaku public Spack setup not found: $setup"
  bash -c '. "$1" && spack location -i "/$2"' _ "$setup" "$1"
}

write_ascent_compute_env() {
  local target_python="$1" target_numpy="$2" target_mpi4py="$3"
  local adios_lib=""
  if [[ -d "$STACK_ADIOS2/lib" ]]; then
    adios_lib=":$STACK_ADIOS2/lib"
  fi
  cat >"$STACK_ASCENT/compute-env.sh" <<EOF
# Source on a Fugaku compute node after loading the matching LLVM module.
# This environment uses aarch64 Python; do not source the login-node env.sh.
export PYTHONHOME="$target_python:$target_python"
export PATH="$target_python/bin:\$PATH"
export PYTHONPATH="$STACK_ASCENT/python-modules:$target_numpy/lib/python3.11/site-packages:$target_mpi4py/lib/python3.11/site-packages:$REPO_ROOT/python/src\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="$STACK_ASCENT/ascent-checkout/lib:$STACK_ASCENT/conduit-v0.9.5/lib:$target_python/lib$adios_lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
}

install_ascent_component() {
  if [[ "$WITH_ASCENT" != true ]]; then
    return 0
  fi
  if [[ "$IS_CROSS" == true && "$ASCENT_EXTRACTS_ONLY" != true ]]; then
    die "--with-ascent requires a native build (cross-compilation cache detected); rerun without --with-ascent"
  fi
  local py_tag
  local profile="slim" target_python="" target_numpy="" target_mpi4py="" host_python=""
  if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
    profile="extracts"
    # Site-provided Python/NumPy/mpi4py share the same aarch64 Python 3.11.
    # Environment overrides allow using another compatible public installation.
    target_python="${PICNIX_ASCENT_TARGET_PYTHON_PREFIX:-$(spack_public_prefix 6pchiok)}"
    target_numpy="${PICNIX_ASCENT_TARGET_NUMPY_PREFIX:-$(spack_public_prefix irn3kud)}"
    target_mpi4py="${PICNIX_ASCENT_TARGET_MPI4PY_PREFIX:-$(spack_public_prefix qx6sbio)}"
    host_python="${PICNIX_ASCENT_HOST_PYTHON_PREFIX:-$(spack_public_prefix k6mf2vt)}"
    py_tag="${target_python}:${target_numpy}:${target_mpi4py}:${host_python}"
  else
    py_tag="$(venv_python_tag)"
    if [[ "$ASCENT_FULL" == true ]]; then
      profile="full"
    fi
  fi
  local stamp_items=(
    "script:$REPO_ROOT/scripts/install_ascent.sh"
    "python_is_venv=true"
    "profile=$profile"
  )
  if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
    stamp_items+=("target_python=$py_tag")
  fi
  local expected
  expected="$(compute_stamp ascent "${stamp_items[@]}")"
  if [[ "$FORCE" == true ]]; then
    clear_stamp ascent
  fi
  # Conduit/Ascent extensions are ABI-tied to the interpreter.
  if [[ -f "$STAMP_DIR/ascent.meta" && -n "$py_tag" ]]; then
    local old_py
    old_py="$(awk -F= '/^venv_python=/ {print $2}' "$STAMP_DIR/ascent.meta" 2>/dev/null || true)"
    if [[ -n "$old_py" && "$old_py" != "$py_tag" ]]; then
      log "Ascent Python ABI changed ($old_py -> $py_tag); invalidating stamp"
      clear_stamp ascent
    fi
  fi
  if [[ -d "$STACK_ASCENT" ]] && stamp_matches ascent "$expected"; then
    log "Ascent up to date: $STACK_ASCENT"
    write_stamp_meta ascent "venv_python=$py_tag" "profile=$profile"
    if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
      printf 'target_python=%s\n' "$py_tag" >>"$STAMP_DIR/ascent.meta"
      write_ascent_compute_env "$target_python" "$target_numpy" "$target_mpi4py"
    fi
    return 0
  fi
  log "--- Installing Ascent into $STACK_ASCENT ($profile) ---"
  apply_cache_compiler_env
  local ascent_args=("$STACK_ASCENT")
  if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
    local host_venv="$STACK_ASCENT/build-python-venv"
    mkdir -p "$STACK_ASCENT"
    uv venv "$host_venv" --python "$host_python/bin/python3.11" --seed
    uv pip install --python "$host_venv/bin/python" pip 'numpy==1.26.4'
    ascent_args+=(--python "$host_venv/bin/python" --python-is-venv --slim
      --cross-python-extracts --cache "$CACHE_FILE"
      --target-python "$target_python" --target-numpy "$target_numpy")
  elif [[ "$ASCENT_FULL" == true ]]; then
    ascent_args+=(--python "$STACK_VENV_BIN" --python-is-venv)
    ascent_args+=(--full)
  else
    ascent_args+=(--python "$STACK_VENV_BIN" --python-is-venv)
    ascent_args+=(--slim)
  fi
  MPICC="$MPICC_EXECUTABLE" MPICXX="$MPICXX_EXECUTABLE" \
    "$REPO_ROOT/scripts/install_ascent.sh" "${ascent_args[@]}"
  write_stamp ascent "$expected"
  write_stamp_meta ascent "venv_python=$py_tag" "profile=$profile"
  if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
    printf 'target_python=%s\n' "$py_tag" >>"$STAMP_DIR/ascent.meta"
    write_ascent_compute_env "$target_python" "$target_numpy" "$target_mpi4py"
  fi
}

ld_path_parts() {
  local parts=()
  [[ -d "$STACK_ADIOS2/lib" ]] && parts+=("$STACK_ADIOS2/lib")
  [[ -d "$STACK_ADIOS2/lib64" ]] && parts+=("$STACK_ADIOS2/lib64")
  if [[ -d "$STACK_ASCENT" ]]; then
    local d
    for d in \
      "$STACK_ASCENT/ascent-checkout/lib" \
      "$STACK_ASCENT"/conduit-*/lib
    do
      [[ -d "$d" ]] && parts+=("$d")
    done
  fi
  if (( ${#parts[@]} > 0 )); then
    local IFS=':'
    printf '%s' "${parts[*]}"
  fi
}

suggest_cmake_line() {
  local prefix_paths=()
  [[ "$WITH_DEPS" == true ]] && prefix_paths+=("$STACK_DEPS")
  [[ "$WITH_ADIOS2" == true ]] && prefix_paths+=("$STACK_ADIOS2")
  [[ "$WITH_ASCENT" == true ]] && prefix_paths+=("$STACK_ASCENT")
  local prefix=""
  if (( ${#prefix_paths[@]} > 0 )); then
    local IFS=':'
    prefix="${prefix_paths[*]}"
  fi
  local cache_args=""
  if [[ -n "$CACHE_FILE" ]]; then
    cache_args=" -C ${CACHE_FILE#"$REPO_ROOT"/}"
  fi
  local line="cmake -S . -B build${cache_args} \\"
  line+=$'\n'"  -DCMAKE_BUILD_TYPE=Release \\"
  if [[ -n "$prefix" ]]; then
    line+=$'\n'"  -DCMAKE_PREFIX_PATH=\"$prefix\" \\"
  fi
  line+=$'\n'"  -DPICNIX_USE_SYSTEM_LIBS=ON"
  if [[ "$WITH_ADIOS2" == true ]]; then
    line+=$' \\\n'"  -DPICNIX_ENABLE_ADIOS2=ON -DPICNIX_ADIOS2_ROOT=$STACK_ADIOS2"
  fi
  if [[ "$WITH_ASCENT" == true ]]; then
    line+=$' \\\n'"  -DPICNIX_ENABLE_ASCENT=ON -DPICNIX_ASCENT_ROOT=$STACK_ASCENT"
  fi
  printf '%s' "$line"
}

write_env_sh() {
  local ld_path
  ld_path="$(ld_path_parts)"
  local suggested
  suggested="$(suggest_cmake_line)"
  cat >"$ENV_SH" <<EOF
# Generated by scripts/prepare_build_stack.sh -- do not edit.
# Put site-specific overrides in env.local.sh (sourced at the end if present).
# Do not move this directory: CMake caches and stamps store absolute paths.
#
# Usage (build, analysis, and job scripts):
#   source $ENV_SH
#
# Sets PATH/VIRTUAL_ENV without sourcing venv activate, so your shell prompt
# does not change. For native builds, source this in job scripts too.
# For Fugaku cross-built Ascent Python extracts, use ascent/compute-env.sh
# in the job instead: this login-node venv cannot run on compute nodes.
#
# Compiler fingerprint: $COMPILER_FINGERPRINT
# MPI C wrapper:   $MPICC_EXECUTABLE
# MPI C++ wrapper: $MPICXX_EXECUTABLE

export PICNIX_STACK="$STACK_DIR"

# Drop a different venv: restore saved PATH if we have it, otherwise strip
# the foreign venv's bin dir so it cannot linger behind the stack prefix.
if [ -n "\${VIRTUAL_ENV:-}" ] && [ "\$VIRTUAL_ENV" != "$STACK_PYTHON" ]; then
  if [ -n "\${_PICNIX_OLD_PATH:-}" ]; then
    PATH="\$_PICNIX_OLD_PATH"
    export PATH
    unset _PICNIX_OLD_PATH
  else
    _foreign_bin="\$VIRTUAL_ENV/bin"
    PATH=":\$PATH:"
    PATH="\${PATH//:\$_foreign_bin:/:}"
    PATH="\${PATH#:}"
    PATH="\${PATH%:}"
    export PATH
    unset _foreign_bin
  fi
  unset VIRTUAL_ENV
fi

# Prefer the stack interpreter without running activate (no PS1/prompt change).
if [ -d "$STACK_PYTHON/bin" ]; then
  case ":\$PATH:" in
    *":$STACK_PYTHON/bin:"*) ;;
    *)
      if [ -z "\${_PICNIX_OLD_PATH:-}" ] && [ -n "\${VIRTUAL_ENV:-}" ]; then
        _PICNIX_OLD_PATH="\$PATH"
        export _PICNIX_OLD_PATH
      fi
      PATH="$STACK_PYTHON/bin:\$PATH"
      export PATH
      ;;
  esac
  export VIRTUAL_ENV="$STACK_PYTHON"
fi

export MPICC="$MPICC_EXECUTABLE"
export MPICXX="$MPICXX_EXECUTABLE"

STACK_PREFIX_PATH=""
EOF

  {
    if [[ "$WITH_DEPS" == true ]]; then
      printf 'STACK_PREFIX_PATH="${STACK_PREFIX_PATH:+$STACK_PREFIX_PATH:}%s"\n' "$STACK_DEPS"
    fi
    if [[ "$WITH_ADIOS2" == true ]]; then
      printf 'STACK_PREFIX_PATH="${STACK_PREFIX_PATH:+$STACK_PREFIX_PATH:}%s"\n' "$STACK_ADIOS2"
      printf 'export PICNIX_ADIOS2_ROOT="%s"\n' "$STACK_ADIOS2"
    else
      printf '# PICNIX_ADIOS2_ROOT is unset; add --with-adios2 to enable ADIOS2.\n'
    fi
    if [[ "$WITH_ASCENT" == true ]]; then
      printf 'STACK_PREFIX_PATH="${STACK_PREFIX_PATH:+$STACK_PREFIX_PATH:}%s"\n' "$STACK_ASCENT"
      printf 'export PICNIX_ASCENT_ROOT="%s"\n' "$STACK_ASCENT"
    else
      printf '# PICNIX_ASCENT_ROOT is unset; add --with-ascent to enable Ascent.\n'
    fi
  } >>"$ENV_SH"

  cat >>"$ENV_SH" <<EOF

if [ -n "\$STACK_PREFIX_PATH" ]; then
  export CMAKE_PREFIX_PATH="\$STACK_PREFIX_PATH\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}"
fi

EOF

  if [[ -n "$ld_path" ]]; then
    printf 'export LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n\n' "$ld_path" >>"$ENV_SH"
  fi

  local suggested commented
  suggested="$(suggest_cmake_line)"
  commented="$(printf '%s\n' "$suggested" | sed 's/^/# /')"

  cat >>"$ENV_SH" <<EOF
# Suggested PIC-NIX configure (run from the repository root):
#
$commented

if [ -f "$ENV_LOCAL_SH" ]; then
  . "$ENV_LOCAL_SH"
fi
EOF

  if [[ ! -f "$ENV_LOCAL_SH" ]]; then
    cat >"$ENV_LOCAL_SH" <<'EOF'
# Site-specific overrides for this build stack (sourced by env.sh).
# Example:
#   export OMP_NUM_THREADS=4
EOF
  fi
  log "Wrote $ENV_SH"
}

print_summary() {
  local suggested
  suggested="$(suggest_cmake_line)"
  cat <<EOF

Build stack ready: $STACK_DIR

  Python:     $STACK_VENV_BIN
  C++ deps:   $([[ "$WITH_DEPS" == true ]] && echo "$STACK_DEPS" || echo "(skipped)")
  ADIOS2:     $([[ "$WITH_ADIOS2" == true ]] && echo "$STACK_ADIOS2 (python=$([[ "$WITH_ADIOS2_PYTHON" == true ]] && echo on || echo off))" || echo "(not built; --with-adios2)")
  Ascent:     $([[ "$WITH_ASCENT" == true ]] && echo "$STACK_ASCENT" || echo "(not built; --with-ascent)")
  Compiler:   $MPICXX_EXECUTABLE
  Fingerprint:$COMPILER_FINGERPRINT

Activate with:

  source $ENV_SH

Suggested configure:

$suggested

EOF
}

check_stack() {
  local failed=0
  log "--- Checking stack: $STACK_DIR ---"
  if [[ ! -f "$ENV_SH" ]]; then
    err "missing $ENV_SH"
    failed=1
  fi
  if [[ ! -x "$STACK_VENV_BIN" ]]; then
    err "missing virtual environment: $STACK_VENV_BIN"
    failed=1
  else
    local mod
    local python_modules=(numpy)
    if [[ "$IS_CROSS" != true ]]; then
      python_modules+=(mpi4py)
    fi
    for mod in "${python_modules[@]}"; do
      if ! "$STACK_VENV_BIN" -c "import $mod" >/dev/null 2>&1; then
        err "python module not importable: $mod"
        failed=1
      fi
    done
    # picnix is optional; stamp/meta records whether it was requested.
    if ! "$STACK_VENV_BIN" -c 'import picnix' >/dev/null 2>&1; then
      local want_picnix=""
      if [[ -f "$STAMP_DIR/python.meta" ]]; then
        want_picnix="$(awk -F= '/^with_picnix=/ {print $2}' "$STAMP_DIR/python.meta" 2>/dev/null || true)"
      fi
      if [[ "$want_picnix" == "true" ]]; then
        err "python module not importable: picnix (with_picnix=true)"
        failed=1
      else
        log "note: picnix not installed in stack venv (ok with --no-picnix)"
      fi
    fi
  fi

  resolve_compiler
  build_fingerprint

  local name path
  for name in python deps adios2 ascent; do
    path="$(stamp_path "$name")"
    if [[ -f "$path" ]]; then
      log "stamp $name: present"
    else
      log "stamp $name: absent (component not built or skipped)"
    fi
  done

  # Recompute stamps that embed the compiler fingerprint / script hashes so
  # a changed cache or installer invalidates optional components too.
  if [[ -f "$(stamp_path deps)" ]]; then
    local dep_items=()
    if [[ -f "$STAMP_DIR/deps.meta" ]]; then
      mapfile -t dep_items <"$STAMP_DIR/deps.meta"
    else
      dep_items=("script:$REPO_ROOT/scripts/install_dependencies.sh")
      if [[ -n "$CACHE_FILE" ]]; then
        dep_items+=("cache:$CACHE_FILE")
      fi
    fi
    if (( ${#dep_items[@]} > 0 )); then
      local expected
      expected="$(compute_stamp deps "${dep_items[@]}")"
      if [[ "$(cat "$(stamp_path deps)")" != "$expected" ]]; then
        err "deps stamp does not match current compiler/cache fingerprint (rerun with --force if intended)"
        failed=1
      fi
    fi
  fi

  if [[ -f "$(stamp_path adios2)" && -f "$STAMP_DIR/adios2.meta" ]]; then
    local adios_mode
    adios_mode="$(awk -F= '/^python_mode=/ {print $2}' "$STAMP_DIR/adios2.meta" 2>/dev/null || true)"
    if [[ -n "$adios_mode" ]]; then
      local expected
      expected="$(compute_stamp adios2 \
        "script:$REPO_ROOT/scripts/install_adios2.sh" \
        "python_mode=$adios_mode")"
      if [[ "$(cat "$(stamp_path adios2)")" != "$expected" ]]; then
        err "adios2 stamp does not match current fingerprint/script (rerun with --force if intended)"
        failed=1
      fi
    fi
  fi

  if [[ -f "$(stamp_path ascent)" && -f "$STAMP_DIR/ascent.meta" ]]; then
    local ascent_profile
    ascent_profile="$(awk -F= '/^profile=/ {print $2}' "$STAMP_DIR/ascent.meta" 2>/dev/null || true)"
    if [[ -n "$ascent_profile" ]]; then
      local ascent_items=(
        "script:$REPO_ROOT/scripts/install_ascent.sh"
        "python_is_venv=true"
        "profile=$ascent_profile"
      )
      if [[ "$ascent_profile" == "extracts" ]]; then
        local target_tag
        target_tag="$(awk -F= '/^target_python=/ {print $2}' "$STAMP_DIR/ascent.meta" 2>/dev/null || true)"
        ascent_items+=("target_python=$target_tag")
      fi
      local expected
      expected="$(compute_stamp ascent "${ascent_items[@]}")"
      if [[ "$(cat "$(stamp_path ascent)")" != "$expected" ]]; then
        err "ascent stamp does not match current fingerprint/script (rerun with --force if intended)"
        failed=1
      fi
    fi
  fi

  if [[ -f "$(stamp_path python)" && -f "$STAMP_DIR/python.meta" ]]; then
    local want_picnix want_adios want_adios_py want_ascent want_full desired_py packages
    want_picnix="$(awk -F= '/^with_picnix=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    want_adios="$(awk -F= '/^with_adios2=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    want_adios_py="$(awk -F= '/^with_adios2_python=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    want_ascent="$(awk -F= '/^with_ascent=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    want_full="$(awk -F= '/^ascent_full=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    desired_py="$(awk -F= '/^desired_python=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    packages="$(awk -F= '/^packages=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
    if [[ -n "$desired_py" && -n "$packages" ]]; then
      local py_items=(
        "with_picnix=$want_picnix"
        "with_adios2=$want_adios"
        "with_adios2_python=$want_adios_py"
        "with_ascent=$want_ascent"
        "ascent_full=$want_full"
        "desired_python=$desired_py"
        "packages=$packages"
      )
      local pyproject_hash
      pyproject_hash="$(awk -F= '/^pyproject_hash=/ {print $2}' "$STAMP_DIR/python.meta" || true)"
      if [[ -n "$pyproject_hash" ]]; then
        py_items+=("pyproject_hash=$pyproject_hash")
      fi
      local expected
      expected="$(compute_stamp python "${py_items[@]}")"
      if [[ "$(cat "$(stamp_path python)")" != "$expected" ]]; then
        err "python stamp does not match current fingerprint/meta (rerun with --force if intended)"
        failed=1
      fi
    fi
  fi

  if [[ -d "$STACK_ADIOS2" ]]; then
    if ! compgen -G "$STACK_ADIOS2/lib*/cmake/adios2" >/dev/null; then
      err "ADIOS2 CMake package missing under $STACK_ADIOS2"
      failed=1
    fi
    if [[ -x "$STACK_VENV_BIN" ]] && "$STACK_VENV_BIN" -c 'import adios2' >/dev/null 2>&1; then
      log "python import adios2: ok"
    else
      log "python import adios2: not in stack venv (expected for C++-only ADIOS2; use python[adios] to read)"
    fi
  fi

  if [[ -d "$STACK_ASCENT" ]]; then
    if [[ ! -f "$STACK_ASCENT/ascent-checkout/lib/cmake/ascent/AscentConfig.cmake" ]]; then
      err "AscentConfig.cmake missing under $STACK_ASCENT"
      failed=1
    fi
    if [[ -f "$STAMP_DIR/ascent.meta" ]] && grep -q '^profile=extracts$' "$STAMP_DIR/ascent.meta"; then
      for path in "$STACK_ASCENT/python-modules/conduit/conduit_python.so" \
                  "$STACK_ASCENT/python-modules/ascent/mpi/ascent_mpi_python.so" \
                  "$STACK_ASCENT/compute-env.sh"; do
        if [[ ! -f "$path" ]]; then
          err "missing Ascent cross-build artifact: $path"
          failed=1
        fi
      done
    elif [[ -x "$STACK_VENV_BIN" ]] && ! "$STACK_VENV_BIN" -c 'import conduit' >/dev/null 2>&1; then
      err "conduit not importable from stack venv"
      failed=1
    fi
  fi

  if (( failed > 0 )); then
    die "stack check failed"
  fi
  log "Stack check passed."
  log "Activate with: source $ENV_SH"
}

main() {
  if [[ "$CHECK_ONLY" == true ]]; then
    [[ -d "$STACK_DIR" ]] || die "stack directory not found: $STACK_DIR"
    # Recover the cache used at install time so the fingerprint (and stamp
    # recompute) match when the user omits --cache.
    if [[ -z "$CACHE_FILE" && -f "$STAMP_DIR/deps.meta" ]]; then
      local meta_cache
      meta_cache="$(sed -n 's/^cache://p' "$STAMP_DIR/deps.meta" | head -n 1)"
      if [[ -n "$meta_cache" && -f "$meta_cache" ]]; then
        CACHE_FILE="$meta_cache"
        log "Using cache from deps.meta: $CACHE_FILE"
      fi
    fi
    if [[ -n "$CACHE_FILE" || -n "$MPICC_EXPLICIT" || -n "$MPICXX_EXPLICIT" ]]; then
      resolve_compiler
      build_fingerprint
    elif [[ -f "$ENV_SH" ]]; then
      MPICC_EXPLICIT="${MPICC:-}"
      MPICXX_EXPLICIT="${MPICXX:-}"
      if [[ -z "$MPICXX_EXPLICIT" ]]; then
        MPICXX_EXPLICIT="mpicxx"
      fi
      if [[ -z "$MPICC_EXPLICIT" ]]; then
        MPICC_EXPLICIT="mpicc"
      fi
      resolve_compiler
      build_fingerprint
    else
      die "--check needs --cache or MPICC/MPICXX (or a prior env.sh with wrappers on PATH)"
    fi
    check_stack
    return 0
  fi

  require_command cmake
  require_command git
  ensure_lock

  resolve_compiler
  build_fingerprint

  if [[ "$ASCENT_EXTRACTS_ONLY" == true ]]; then
    if [[ "$IS_CROSS" != true || -z "$CACHE_FILE" || "$(cache_system_processor)" != "aarch64" ]]; then
      die "--ascent-extracts-only requires a Fugaku aarch64 cross-compilation cache"
    fi
    if [[ "$ASCENT_FULL" == true ]]; then
      die "--ascent-full cannot be combined with --ascent-extracts-only"
    fi
  fi
  if [[ "$IS_CROSS" == true ]]; then
    if [[ ( "$WITH_ASCENT" == true && "$ASCENT_EXTRACTS_ONLY" != true ) || "$WITH_ADIOS2_PYTHON" == true ]]; then
      die "cross-compilation cache detected; --with-ascent needs --ascent-extracts-only, and --with-adios2-python requires a native build"
    fi
    if [[ "$WITH_ADIOS2" == true && ( -z "$CACHE_FILE" || "$(cache_system_processor)" != "aarch64" ) ]]; then
      die "cross-built ADIOS2 requires an aarch64 CMake cache (FFS float format is target-specific)"
    fi
    log "--- Cross-compilation stack (host Python, target C++ libraries) ---"
  fi

  log "--- Build stack: $STACK_DIR ---"
  log "Compiler fingerprint: $COMPILER_FINGERPRINT"
  log "MPI C: $MPICC_EXECUTABLE"
  log "MPI C++: $MPICXX_EXECUTABLE"
  log "Jobs: $JOBS"

  mkdir -p "$STAMP_DIR"
  prepare_python
  install_deps_component
  install_adios2_component
  install_ascent_component
  wire_python_paths
  write_env_sh
  print_summary
}

main "$@"
