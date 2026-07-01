#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
install -m 0755 "${repo_root}/scripts/git-hooks/pre-commit" "${repo_root}/.git/hooks/pre-commit"
echo "Installed pre-commit hook to .git/hooks/pre-commit"
