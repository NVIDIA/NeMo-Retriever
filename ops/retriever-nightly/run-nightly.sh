#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

umask 077

readonly EXIT_CONFIG=64
readonly EXIT_FETCH=69
readonly EXIT_INTERNAL=70
readonly EXIT_CANNOT_CREATE=73
readonly EXIT_ALREADY_RUNNING=75

log() {
    printf 'retriever-nightly: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage: run-nightly.sh [OPTIONS] [--] [RUNFILE ...]

Fetch and run the library and ViDoRe v3 harness suite from the latest
upstream/main commit. With no RUNFILE arguments, all 12 checked-in nightly
runfiles are executed.

Options:
  --ref REF             Run an existing local Git ref instead of fetching main.
  --dataset-paths PATH  Machine-local dataset path map.
  --artifact-root PATH  Parent directory for timestamped session artifacts.
  --check-vidore-access Validate authenticated ViDoRe evaluation-data access and exit.
  --dry-run             Resolve and validate the suite without executing it.
  --no-slack            Do not post, even when SLACK_WEBHOOK_URL is configured.
  --help                Show this help text.
EOF
}

absolute_path() {
    realpath -m -- "$1"
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
        find "$checkout_root" -mindepth 1 -maxdepth 1 -type d -name 'commit-*' -printf '%T@ %p\n' \
            | sort -nr \
            | sed -E 's/^[^ ]+ //'
    )
    for candidate in "${candidates[@]}"; do
        base="$(basename -- "$candidate")"
        if [[ ! "$base" =~ ^commit-[0-9a-f]{40,64}$ ]]; then
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

select_checkout_and_run() {
    local requested_ref="$1"
    shift

    local repository source_name source_ref remote_ref fetched_ref checkout_root keep_count uv_environment
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
    if [[ ! -d "$repository/.git" && ! -f "$repository/.git" ]]; then
        log "controller checkout is not a Git worktree: $repository"
        return "$EXIT_CONFIG"
    fi
    if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=normal)" ]]; then
        log "controller checkout has local changes: $repository"
        return "$EXIT_CONFIG"
    fi
    if ! mkdir -p "$checkout_root"; then
        log "could not create managed checkout root: $checkout_root"
        return "$EXIT_CONFIG"
    fi
    chmod 700 "$checkout_root"

    exec 8>"$checkout_root/.selection.lock" || return "$EXIT_CONFIG"
    if ! flock -n 8; then
        log "another nightly Git selection is already running"
        return "$EXIT_ALREADY_RUNNING"
    fi

    local target_commit selection_label
    if [[ -z "$requested_ref" ]]; then
        if ! git check-ref-format "$remote_ref"; then
            log "RETRIEVER_LATEST_REF is not a valid Git ref: $source_ref"
            return "$EXIT_CONFIG"
        fi
        log "fetching $source_name $source_ref"
        if ! git -C "$repository" fetch --no-tags "$source_name" "+$remote_ref:$fetched_ref"; then
            log "fetch failed; refusing to run a stale commit"
            return "$EXIT_FETCH"
        fi
        target_commit="$(git -C "$repository" rev-parse --verify "$fetched_ref^{commit}")" || \
            return "$EXIT_FETCH"
        selection_label="$source_name/$source_ref"
    else
        if [[ "$requested_ref" == -* ]]; then
            log "--ref must name a Git ref or commit"
            return "$EXIT_CONFIG"
        fi
        target_commit="$(git -C "$repository" rev-parse --verify "$requested_ref^{commit}" 2>/dev/null)" || {
            log "could not resolve --ref $requested_ref to a local commit"
            return "$EXIT_CONFIG"
        }
        selection_label="$requested_ref"
    fi

    local selected_checkout selected_head target_launcher preflight_access run_rc
    selected_checkout="$checkout_root/commit-$target_commit"
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
    log "selected $selection_label commit $target_commit in $selected_checkout"

    preflight_access=1
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--dry-run" || "$arg" == "--check-vidore-access" ]]; then
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

for argument in "$@"; do
    if [[ "$argument" == "--" ]]; then
        break
    fi
    if [[ "$argument" == "--help" ]]; then
        usage
        exit 0
    fi
done

default_nightly_root="$HOME"
raid_nightly_root="/raid/$(id -un)"
if [[ -d "$raid_nightly_root" && -w "$raid_nightly_root" ]]; then
    default_nightly_root="$raid_nightly_root"
fi
nightly_root="${RETRIEVER_NIGHTLY_ROOT:-$default_nightly_root}"
readonly config_file="${RETRIEVER_CONFIG_FILE:-$nightly_root/.config/nemo-retriever/nightly/nightly.env}"
if [[ -f "$config_file" ]]; then
    if [[ "$(stat -c '%a' "$config_file")" != "600" || "$(stat -c '%u' "$config_file")" != "$(id -u)" ]]; then
        log "nightly configuration must be owned by the invoking user with mode 600"
        exit "$EXIT_CONFIG"
    fi
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a
fi
slack_webhook_url="${SLACK_WEBHOOK_URL:-}"
unset SLACK_WEBHOOK_URL

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RETRIEVER_SELECTED_CHECKOUT:-}" ]]; then
    requested_ref=""
    forwarded_args=()
    while (($#)); do
        case "$1" in
            --ref)
                if (($# < 2)); then
                    log "--ref requires a Git ref or commit"
                    exit "$EXIT_CONFIG"
                fi
                if [[ -n "$requested_ref" ]]; then
                    log "--ref may be specified only once"
                    exit "$EXIT_CONFIG"
                fi
                requested_ref="$2"
                shift 2
                ;;
            --)
                forwarded_args+=("$@")
                break
                ;;
            *)
                forwarded_args+=("$1")
                shift
                ;;
        esac
    done
    export RETRIEVER_CONFIG_FILE="$config_file"
    export RETRIEVER_NIGHTLY_ROOT="$nightly_root"
    if [[ -z "${VLLM_DEEP_GEMM_WARMUP+x}" ]]; then
        export VLLM_DEEP_GEMM_WARMUP=skip
    fi
    select_checkout_and_run "$requested_ref" "${forwarded_args[@]}"
    exit $?
fi

checkout="${RETRIEVER_SELECTED_CHECKOUT:-${RETRIEVER_CHECKOUT:-$(git -C "$script_dir" rev-parse --show-toplevel)}}"
nightly_root="${RETRIEVER_NIGHTLY_ROOT:-$nightly_root}"
artifact_root="${RETRIEVER_ARTIFACT_ROOT:-$nightly_root/retriever-nightly-artifacts}"
dataset_paths="${RETRIEVER_DATASET_PATHS:-$checkout/ops/retriever-nightly/dataset_paths.datasets.yaml}"
session_name="${RETRIEVER_SESSION_NAME:-library_vidore_v3_batch_hf}"
slack_title="${RETRIEVER_SLACK_TITLE:-nemo-retriever library + ViDoRe v3 nightly}"
run_mode="${RETRIEVER_MODE:-batch}"
dry_run=0
check_vidore_access=0
skip_slack=0
runfiles=()

while (($#)); do
    case "$1" in
        --dataset-paths)
            if (($# < 2)); then
                log "--dataset-paths requires a path"
                exit "$EXIT_CONFIG"
            fi
            dataset_paths="$2"
            shift 2
            ;;
        --artifact-root)
            if (($# < 2)); then
                log "--artifact-root requires a path"
                exit "$EXIT_CONFIG"
            fi
            artifact_root="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --check-vidore-access)
            check_vidore_access=1
            shift
            ;;
        --no-slack)
            skip_slack=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            runfiles+=("$@")
            break
            ;;
        -*)
            log "unknown option: $1"
            exit "$EXIT_CONFIG"
            ;;
        *)
            runfiles+=("$1")
            shift
            ;;
    esac
done

if ((check_vidore_access && (dry_run || ${#runfiles[@]} > 0))); then
    log "--check-vidore-access cannot be combined with --dry-run or runfiles"
    exit "$EXIT_CONFIG"
fi

if [[ -z "${VLLM_DEEP_GEMM_WARMUP+x}" ]]; then
    export VLLM_DEEP_GEMM_WARMUP=skip
fi

checkout="$(absolute_path "$checkout")"
artifact_root="$(absolute_path "$artifact_root")"
dataset_paths="$(absolute_path "$dataset_paths")"

if [[ -n "${RETRIEVER_UV_BIN:-}" ]]; then
    uv_bin="$RETRIEVER_UV_BIN"
elif uv_bin="$(command -v uv 2>/dev/null)"; then
    :
else
    uv_bin="$HOME/.local/bin/uv"
fi
if [[ "$uv_bin" != /* ]]; then
    uv_bin="$(command -v -- "$uv_bin" 2>/dev/null || absolute_path "$uv_bin")"
fi

if [[ ! -x "$uv_bin" ]]; then
    log "uv executable is missing or not executable: $uv_bin"
    exit "$EXIT_CONFIG"
fi
if [[ ! -d "$checkout/.git" && ! -f "$checkout/.git" ]]; then
    log "deployment checkout is not a Git worktree: $checkout"
    exit "$EXIT_CONFIG"
fi
if [[ -n "$(git -C "$checkout" status --porcelain --untracked-files=normal --ignore-submodules)" ]]; then
    log "deployment checkout has tracked, staged, or untracked changes; refusing to run"
    exit "$EXIT_CONFIG"
fi
if ((check_vidore_access)); then
    cd "$checkout" || exit "$EXIT_CONFIG"
    "$uv_bin" run --frozen --project nemo_retriever retriever harness check-vidore-access
    exit $?
fi
if [[ ! -f "$dataset_paths" ]]; then
    log "dataset path map is missing: $dataset_paths"
    exit "$EXIT_CONFIG"
fi
post_slack=0
if ((!dry_run && !skip_slack)) && [[ -n "$slack_webhook_url" ]]; then
    post_slack=1
    if [[ "$slack_webhook_url" == *$'\n'* || \
        "$slack_webhook_url" != https://hooks.slack.com/services/* ]]; then
        log "SLACK_WEBHOOK_URL must contain one Slack incoming-webhook URL"
        exit "$EXIT_CONFIG"
    fi
fi
if ((${#runfiles[@]} == 0)); then
    runfiles=(
        nemo_retriever/harness/runfiles/jp20_beir.json
        nemo_retriever/harness/runfiles/bo767_beir.json
        nemo_retriever/harness/runfiles/earnings_beir.json
        nemo_retriever/harness/runfiles/financebench_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_computer_science_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_energy_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_finance_en_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_finance_fr_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_hr_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_industrial_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_pharmaceuticals_beir.json
        nemo_retriever/harness/runfiles/vidore_v3_physics_beir.json
    )
fi
readonly -a runfiles
for runfile in "${runfiles[@]}"; do
    resolved_runfile="$runfile"
    if [[ "$resolved_runfile" != /* ]]; then
        resolved_runfile="$checkout/$resolved_runfile"
    fi
    if [[ ! -f "$resolved_runfile" ]]; then
        log "required runfile is missing: $runfile"
        exit "$EXIT_CONFIG"
    fi
done

if ! mkdir -p "$artifact_root"; then
    log "could not create artifact root: $artifact_root"
    exit "$EXIT_CANNOT_CREATE"
fi

exec 9>"$artifact_root/.retriever-nightly.lock" || exit "$EXIT_CANNOT_CREATE"
if ! flock -n 9; then
    log "another nightly invocation holds the lock"
    exit "$EXIT_ALREADY_RUNNING"
fi

readonly timestamp="$(date -u +%Y%m%d_%H%M%S_UTC)"
readonly session_dir="$artifact_root/${session_name}-${timestamp}"
readonly session_summary="$session_dir/session_summary.json"
readonly slack_attempt_marker="$session_dir/.slack_post_attempted"

if [[ -e "$session_dir" ]]; then
    log "refusing to reuse an existing session directory: $session_dir"
    exit "$EXIT_CANNOT_CREATE"
fi

cd "$checkout" || exit "$EXIT_CONFIG"

run_rc=0
dry_run_args=()
isolation_args=(--isolate-runs)
if ((dry_run)); then
    dry_run_args=(--dry-run)
    isolation_args=()
fi
"$uv_bin" run --frozen --project nemo_retriever retriever harness run-files \
    --session-name "$session_name" \
    --output-dir "$session_dir" \
    --dataset-paths "$dataset_paths" \
    --mode "$run_mode" \
    "${isolation_args[@]}" \
    "${dry_run_args[@]}" \
    "${runfiles[@]}" || run_rc=$?

if [[ ! -f "$session_summary" ]]; then
    log "run-files did not produce a terminal session summary: $session_summary"
    if ((run_rc != 0)); then
        exit "$run_rc"
    fi
    exit "$EXIT_INTERNAL"
fi

if ((post_slack == 0)); then
    if ((run_rc != 0)); then
        log "session completed without Slack; returning harness exit code $run_rc: $session_dir"
        exit "$run_rc"
    fi
    log "session completed without Slack: $session_dir"
    exit 0
fi

export SLACK_WEBHOOK_URL="$slack_webhook_url"

if ! (set -o noclobber; : >"$slack_attempt_marker") 2>/dev/null; then
    unset SLACK_WEBHOOK_URL
    log "Slack post was already attempted for this session; refusing a duplicate"
    if ((run_rc != 0)); then
        exit "$run_rc"
    fi
    exit "$EXIT_CONFIG"
fi

post_rc=0
"$uv_bin" run --frozen --project nemo_retriever retriever harness post-slack \
    --title "$slack_title" \
    "$session_dir" || post_rc=$?
unset SLACK_WEBHOOK_URL
unset slack_webhook_url

if ((run_rc != 0)); then
    if ((post_rc != 0)); then
        log "harness and Slack post both failed; returning harness exit code $run_rc"
    else
        log "session report posted; returning harness exit code $run_rc"
    fi
    exit "$run_rc"
fi
if ((post_rc != 0)); then
    log "harness passed but Slack post failed with exit code $post_rc"
    exit "$post_rc"
fi

log "session completed and one Slack post succeeded: $session_dir"
