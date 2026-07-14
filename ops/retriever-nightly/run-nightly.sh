#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

umask 077

readonly EXIT_CONFIG=64
readonly EXIT_INTERNAL=70
readonly EXIT_CANNOT_CREATE=73
readonly EXIT_ALREADY_RUNNING=75

log() {
    printf 'retriever-nightly: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage: run-nightly.sh [OPTIONS] [--] [RUNFILE ...]

Run the library and ViDoRe v3 harness suite from the current clean checkout.
With no RUNFILE arguments, all 12 checked-in nightly runfiles are executed.

Options:
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
