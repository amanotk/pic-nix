#!/usr/bin/env bash
set -euo pipefail

if [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]; then
  cat <<'EOF'
  Usage: scripts/install_catch2v3.sh [install_prefix]

Install Catch2 v3.5.4 into the given prefix (default: $HOME/usr).
EOF
  exit 0
fi

prefix="${1:-$HOME/usr}"
repo_dir="${HOME}/tmp/Catch2"

rm -rf "${repo_dir}"
mkdir -p "${repo_dir}"

git clone https://github.com/catchorg/Catch2.git "${repo_dir}"
cd "${repo_dir}"
git checkout v3.5.4

cmake -S . -B build -DCMAKE_INSTALL_PREFIX="${prefix}"
cmake --build build
cmake --install build

rm -rf "${repo_dir}"

echo "Installed Catch2 to ${prefix}"
echo "Config file: ${prefix}/lib/cmake/Catch2/Catch2Config.cmake"
