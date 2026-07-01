#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME=nix
REMOTE_URL=https://github.com/amanotk/nix
SUBTREE_PREFIX=nix/
DEFAULT_BRANCH=develop

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Sync the nix/ subtree with the upstream repository:
  ${REMOTE_URL}

Commands:
  setup    Add the nix remote (one-time)
  fetch    Fetch upstream refs
  pull     Pull upstream branch into nix/ (--squash)
  push     Push nix/ subtree to upstream branch
  log      Show upstream log

Options:
  --branch <name>   Upstream (remote) branch [default: ${DEFAULT_BRANCH}]
                    The local branch is the current pic-nix branch.

The local side is always the current git branch.
--branch only controls which upstream branch to sync with.
EOF
    exit "${1:-0}"
}

parse_args() {
    BRANCH="${DEFAULT_BRANCH}"
    PASSTHROUGH=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --branch)
                [[ $# -lt 2 ]] && { echo "Error: --branch requires an argument" >&2; usage 1; }
                BRANCH="$2"; shift 2
                ;;
            -h|--help)
                usage 0
                ;;
            --)
                shift; PASSTHROUGH+=("$@"); break
                ;;
            *)
                PASSTHROUGH+=("$1"); shift
                ;;
        esac
    done
}

cmd_setup() {
    if git remote get-url "${REMOTE_NAME}" &>/dev/null; then
        echo "Remote '${REMOTE_NAME}' already exists: $(git remote get-url "${REMOTE_NAME}")"
    else
        git remote add "${REMOTE_NAME}" "${REMOTE_URL}"
        echo "Added remote '${REMOTE_NAME}' -> ${REMOTE_URL}"
    fi
}

cmd_fetch() {
    parse_args "$@"
    git fetch "${REMOTE_NAME}"
}

cmd_pull() {
    parse_args "$@"
    echo "Pulling ${REMOTE_NAME}/${BRANCH} into ${SUBTREE_PREFIX} (local branch: $(git branch --show-current))"
    git subtree pull --prefix="${SUBTREE_PREFIX}" "${REMOTE_NAME}" "${BRANCH}" --squash
}

cmd_push() {
    parse_args "$@"
    echo "Pushing ${SUBTREE_PREFIX} to ${REMOTE_NAME}/${BRANCH} (from local branch: $(git branch --show-current))"
    git subtree push --prefix="${SUBTREE_PREFIX}" "${REMOTE_NAME}" "${BRANCH}"
}

cmd_log() {
    parse_args "$@"
    git log "${REMOTE_NAME}/${BRANCH}" "${PASSTHROUGH[@]}"
}

[[ $# -eq 0 ]] || [[ "${1}" == "-h" ]] || [[ "${1}" == "--help" ]] && usage 0

cmd="${1}"; shift
case "${cmd}" in
    setup) cmd_setup "$@" ;;
    fetch) cmd_fetch "$@" ;;
    pull)  cmd_pull  "$@" ;;
    push)  cmd_push  "$@" ;;
    log)   cmd_log   "$@" ;;
    *)     echo "Unknown command: ${cmd}" >&2; usage 1 ;;
esac
