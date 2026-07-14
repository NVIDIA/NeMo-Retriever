#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

umask 077

readonly EXIT_CONFIG=64
readonly EXIT_FETCH=69
readonly EXIT_ALREADY_RUNNING=75

log() {
    printf 'retriever-nightly-latest: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage: run-latest-main.sh [RUN-NIGHTLY-OPTION ...]

Fetch upstream/main into an immutable worktree, preflight ViDoRe access, and
run that commit's nightly launcher. All non-help options are forwarded to
run-nightly.sh.

Options:
  --help  Show this help text without fetching or requiring configuration.
EOF
}

prune_managed_worktrees() {
    local repository="$1"
    local checkout_root="$2"
    local selected_checkout="$3"
    local keep_count="$4"
    local retained=0
    local candidate base
    local -a candidates=()

    mapfile -t candidates < <(
        find "$checkout_root" -mindepth 1 -maxdepth 1 -type d -name 'main-*' -printf '%T@ %p\n' \
            | sort -nr \
            | sed -E 's/^[^ ]+ //'
    )
    for candidate in "${candidates[@]}"; do
        base="$(basename -- "$candidate")"
        if [[ ! "$base" =~ ^main-[0-9a-f]{40,64}$ ]]; then
            continue
        fi
        if [[ "$candidate" == "$selected_checkout" || $retained -lt $keep_count ]]; then
            ((retained += 1))
            continue
        fi
        if [[ -n "$(git -C "$candidate" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
            log "retaining modified managed worktree: $candidate"
            continue
        fi
        if ! git -C "$repository" worktree remove --force "$candidate"; then
            log "could not prune managed worktree: $candidate"
        fi
    done
    git -C "$repository" worktree prune || log "could not prune stale Git worktree metadata"
}

main() {
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--help" ]]; then
            usage
            return 0
        fi
    done

    local default_nightly_root="$HOME"
    local raid_nightly_root="/raid/$(id -un)"
    if [[ -d "$raid_nightly_root" && -w "$raid_nightly_root" ]]; then
        default_nightly_root="$raid_nightly_root"
    fi
    local nightly_root="${RETRIEVER_NIGHTLY_ROOT:-$default_nightly_root}"
    local config_file="${RETRIEVER_CONFIG_FILE:-$nightly_root/.config/nemo-retriever/nightly/nightly.env}"
    if [[ -f "$config_file" ]]; then
        if [[ "$(stat -c '%a' "$config_file")" != "600" || \
            "$(stat -c '%u' "$config_file")" != "$(id -u)" ]]; then
            log "nightly configuration must be owned by the invoking user with mode 600"
            return "$EXIT_CONFIG"
        fi
        set -a
        # shellcheck disable=SC1090
        source "$config_file"
        set +a
    fi
    local slack_webhook_url="${SLACK_WEBHOOK_URL:-}"
    unset SLACK_WEBHOOK_URL

    nightly_root="${RETRIEVER_NIGHTLY_ROOT:-$nightly_root}"
    export RETRIEVER_CONFIG_FILE="$config_file"
    export RETRIEVER_NIGHTLY_ROOT="$nightly_root"
    if [[ -z "${VLLM_DEEP_GEMM_WARMUP+x}" ]]; then
        export VLLM_DEEP_GEMM_WARMUP=skip
    fi

    local script_dir repository source_ref source_name remote_ref fetched_ref checkout_root keep_count uv_environment
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repository="${RETRIEVER_UPDATE_REPOSITORY:-$(git -C "$script_dir" rev-parse --show-toplevel)}"
    source_name="${RETRIEVER_LATEST_SOURCE:-upstream}"
    source_ref="${RETRIEVER_LATEST_REF:-main}"
    remote_ref="$source_ref"
    if [[ "$remote_ref" != refs/* ]]; then
        remote_ref="refs/heads/$remote_ref"
    fi
    fetched_ref="refs/retriever-nightly/latest-main"
    checkout_root="${RETRIEVER_LATEST_CHECKOUT_ROOT:-$nightly_root/retriever-nightly-checkouts}"
    keep_count="${RETRIEVER_LATEST_KEEP_CHECKOUTS:-7}"
    uv_environment="${UV_PROJECT_ENVIRONMENT:-$checkout_root/.venv}"

    if [[ ! "$keep_count" =~ ^[1-9][0-9]*$ ]]; then
        log "RETRIEVER_LATEST_KEEP_CHECKOUTS must be a positive integer"
        return "$EXIT_CONFIG"
    fi
    if ! git check-ref-format "$remote_ref"; then
        log "RETRIEVER_LATEST_REF is not a valid Git ref: $source_ref"
        return "$EXIT_CONFIG"
    fi
    if [[ ! -d "$repository/.git" && ! -f "$repository/.git" ]]; then
        log "update repository is not a Git worktree: $repository"
        return "$EXIT_CONFIG"
    fi
    if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=normal)" ]]; then
        log "reviewed controller checkout has local changes: $repository"
        return "$EXIT_CONFIG"
    fi
    if ! mkdir -p "$checkout_root"; then
        log "could not create latest-main checkout root: $checkout_root"
        return "$EXIT_CONFIG"
    fi
    chmod 700 "$checkout_root"

    exec 8>"$checkout_root/.latest-main.lock" || return "$EXIT_CONFIG"
    if ! flock -n 8; then
        log "another latest-main selection is already running"
        return "$EXIT_ALREADY_RUNNING"
    fi

    log "fetching $source_name $source_ref"
    if ! git -C "$repository" fetch --no-tags "$source_name" "+$remote_ref:$fetched_ref"; then
        log "fetch failed; refusing to run a stale commit"
        return "$EXIT_FETCH"
    fi

    local target_commit selected_checkout selected_head target_launcher preflight_access run_rc
    target_commit="$(git -C "$repository" rev-parse --verify "$fetched_ref^{commit}")" || return "$EXIT_FETCH"
    selected_checkout="$checkout_root/main-$target_commit"
    if [[ -e "$selected_checkout" ]]; then
        selected_head="$(git -C "$selected_checkout" rev-parse --verify HEAD 2>/dev/null || true)"
        if [[ "$selected_head" != "$target_commit" ]]; then
            log "managed checkout has an unexpected commit: $selected_checkout"
            return "$EXIT_CONFIG"
        fi
        if [[ -n "$(git -C "$selected_checkout" status --porcelain --untracked-files=normal)" ]]; then
            log "managed checkout has local changes: $selected_checkout"
            return "$EXIT_CONFIG"
        fi
    elif ! git -C "$repository" worktree add --detach "$selected_checkout" "$target_commit"; then
        log "could not create detached worktree for $target_commit"
        return "$EXIT_CONFIG"
    fi
    touch "$selected_checkout"

    target_launcher="$selected_checkout/ops/retriever-nightly/run-nightly.sh"
    if [[ ! -x "$target_launcher" ]]; then
        log "selected commit does not contain an executable nightly launcher: $target_launcher"
        return "$EXIT_CONFIG"
    fi

    export RETRIEVER_SELECTED_CHECKOUT="$selected_checkout"
    export UV_PROJECT_ENVIRONMENT="$uv_environment"
    log "selected $source_name/$source_ref commit $target_commit in $selected_checkout"

    preflight_access=1
    for arg in "$@"; do
        if [[ "$arg" == "--dry-run" || "$arg" == "--check-vidore-access" || "$arg" == "--help" ]]; then
            preflight_access=0
        fi
    done

    run_rc=0
    if [[ "$preflight_access" == "1" ]]; then
        "$target_launcher" --check-vidore-access || run_rc=$?
    fi
    if ((run_rc == 0)); then
        if [[ -n "$slack_webhook_url" ]]; then
            export SLACK_WEBHOOK_URL="$slack_webhook_url"
        fi
        "$target_launcher" "$@" || run_rc=$?
        unset SLACK_WEBHOOK_URL
    else
        log "ViDoRe access preflight failed; skipping GPU work"
    fi

    prune_managed_worktrees "$repository" "$checkout_root" "$selected_checkout" "$keep_count"
    return "$run_rc"
}

main "$@"
