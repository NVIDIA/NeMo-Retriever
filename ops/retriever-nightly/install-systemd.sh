#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXIT_CONFIG=64
readonly SERVICE_NAME="nrl-harness-batch-hf.service"
readonly TIMER_NAME="nrl-harness-batch-hf.timer"
temporary_directory=""

cleanup() {
    if [[ -n "$temporary_directory" && -d "$temporary_directory" ]]; then
        rm -rf -- "$temporary_directory"
    fi
}

trap cleanup EXIT

log() {
    printf 'retriever-nightly-install: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage: install-systemd.sh [--test-current-branch]
       install-systemd.sh --render DIRECTORY [--test-current-branch]
       install-systemd.sh --uninstall

Render and install a root-managed daily systemd service for the current user
and this clean controller checkout. The production default runs the latest
fetched upstream/main commit.

Options:
  --render DIRECTORY     Render both units without installing or using sudo.
  --test-current-branch  Temporarily schedule the current branch's tracked
                         remote branch for pre-merge deployment validation.
  --uninstall            Disable and remove the installed service and timer.
  --help                 Show this help text.
EOF
}

escape_systemd_value() {
    local value="$1"
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        log "systemd values cannot contain newlines"
        return "$EXIT_CONFIG"
    fi
    value="${value//%/%%}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "$value"
}

escape_systemd_path() {
    local value="$1"
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        log "systemd paths cannot contain newlines"
        return "$EXIT_CONFIG"
    fi
    value="${value//%/%%}"
    value="${value//\\/\\x5c}"
    value="${value//\"/\\x22}"
    value="${value//$'\t'/\\x09}"
    value="${value// /\\x20}"
    printf '%s' "$value"
}

detect_nightly_root() {
    local service_user="$1"
    local service_home="$2"
    local raid_root="/raid/$service_user"
    if [[ -d "$raid_root" && -w "$raid_root" ]]; then
        printf '%s' "$raid_root"
    else
        printf '%s' "$service_home"
    fi
}

render_units() {
    local output_dir="$1"
    local repository="$2"
    local service_user="$3"
    local service_group="$4"
    local service_home="$5"
    local source_name="$6"
    local source_ref="$7"
    local script_dir="$8"
    local escaped_repository escaped_repository_path escaped_home escaped_source escaped_ref
    escaped_repository="$(escape_systemd_value "$repository")"
    escaped_repository_path="$(escape_systemd_path "$repository")"
    escaped_home="$(escape_systemd_value "$service_home")"
    escaped_source="$(escape_systemd_value "$source_name")"
    escaped_ref="$(escape_systemd_value "$source_ref")"

    mkdir -p "$output_dir"
    {
        printf '%s\n' \
            '# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.' \
            '# SPDX-License-Identifier: Apache-2.0' \
            '' \
            '[Unit]' \
            'Description=Run latest selected NeMo Retriever and ViDoRe v3 nightly session'
        printf 'Documentation="file:%s/ops/retriever-nightly/README.md"\n' "$escaped_repository"
        printf '%s\n' \
            'Wants=network-online.target' \
            'After=network-online.target' \
            '' \
            '[Service]' \
            'Type=oneshot'
        printf 'User=%s\n' "$service_user"
        printf 'Group=%s\n' "$service_group"
        printf 'WorkingDirectory=%s\n' "$escaped_repository_path"
        printf 'Environment="HOME=%s"\n' "$escaped_home"
        if [[ -n "$source_name" ]]; then
            printf 'Environment="RETRIEVER_LATEST_SOURCE=%s"\n' "$escaped_source"
            printf 'Environment="RETRIEVER_LATEST_REF=%s"\n' "$escaped_ref"
        fi
        printf 'ExecStart=/usr/bin/bash "%s/ops/retriever-nightly/run-latest-main.sh"\n' "$escaped_repository"
        printf '%s\n' \
            'UMask=0077' \
            'TimeoutStartSec=infinity' \
            'Restart=no'
    } >"$output_dir/$SERVICE_NAME"
    install -m 0644 "$script_dir/systemd/$TIMER_NAME" "$output_dir/$TIMER_NAME"
    chmod 0644 "$output_dir/$SERVICE_NAME"
}

main() {
    local mode="install"
    local action_selected=0
    local render_dir=""
    local test_current_branch=0
    while (($#)); do
        case "$1" in
            --render)
                [[ $# -ge 2 ]] || { usage >&2; return "$EXIT_CONFIG"; }
                ((action_selected == 0)) || { usage >&2; return "$EXIT_CONFIG"; }
                action_selected=1
                mode="render"
                render_dir="$2"
                shift 2
                ;;
            --test-current-branch)
                test_current_branch=1
                shift
                ;;
            --uninstall)
                ((action_selected == 0)) || { usage >&2; return "$EXIT_CONFIG"; }
                action_selected=1
                mode="uninstall"
                shift
                ;;
            --help)
                usage
                return 0
                ;;
            *)
                usage >&2
                return "$EXIT_CONFIG"
                ;;
        esac
    done
    if [[ "$mode" == "uninstall" && $test_current_branch == 1 ]]; then
        log "--uninstall and --test-current-branch cannot be combined"
        return "$EXIT_CONFIG"
    fi
    if [[ "$mode" == "uninstall" ]]; then
        sudo systemctl disable --now "$TIMER_NAME" || true
        sudo rm -f "/etc/systemd/system/$SERVICE_NAME" "/etc/systemd/system/$TIMER_NAME"
        sudo systemctl daemon-reload
        log "removed $SERVICE_NAME and $TIMER_NAME"
        return 0
    fi

    local script_dir repository service_user service_group service_home nightly_root config_file
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
    service_user="$(id -un)"
    service_group="$(id -gn)"
    service_home="${HOME:?HOME must be set}"
    nightly_root="$(detect_nightly_root "$service_user" "$service_home")"
    config_file="$nightly_root/.config/nemo-retriever/nightly/nightly.env"

    if [[ ! "$service_user" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*[$]?$ || \
        ! "$service_group" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*[$]?$ ]]; then
        log "service user and group names must be valid systemd identifiers"
        return "$EXIT_CONFIG"
    fi

    local source_name=""
    local source_ref=""
    if ((test_current_branch)); then
        local branch merge_ref
        branch="$(git -C "$repository" symbolic-ref --quiet --short HEAD)" || {
            log "--test-current-branch requires a checked-out branch"
            return "$EXIT_CONFIG"
        }
        source_name="$(git -C "$repository" config --get "branch.$branch.remote" || true)"
        merge_ref="$(git -C "$repository" config --get "branch.$branch.merge" || true)"
        if [[ -z "$source_name" || "$merge_ref" != refs/heads/* ]]; then
            log "current branch must track a remote branch"
            return "$EXIT_CONFIG"
        fi
        source_ref="${merge_ref#refs/heads/}"
        log "test mode will fetch $source_name/$source_ref instead of upstream/main"
    elif [[ "$mode" != "render" ]] && ! git -C "$repository" remote get-url upstream >/dev/null 2>&1; then
        log "production installation requires an upstream remote"
        return "$EXIT_CONFIG"
    fi

    if [[ "$mode" == "render" ]]; then
        render_units "$render_dir" "$repository" "$service_user" "$service_group" "$service_home" \
            "$source_name" "$source_ref" "$script_dir"
        log "rendered portable units in $render_dir"
        return 0
    fi

    if ((EUID == 0)); then
        log "run this installer as the intended service user, without sudo"
        return "$EXIT_CONFIG"
    fi
    if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=normal)" ]]; then
        log "controller checkout has local changes: $repository"
        return "$EXIT_CONFIG"
    fi
    if [[ ! -f "$config_file" ]]; then
        log "nightly configuration is missing: $config_file"
        return "$EXIT_CONFIG"
    fi
    if [[ "$(stat -c '%a' "$config_file")" != "600" || \
        "$(stat -c '%u' "$config_file")" != "$(id -u)" ]]; then
        log "nightly configuration must be owned by $service_user with mode 600"
        return "$EXIT_CONFIG"
    fi

    temporary_directory="$(mktemp -d)"
    render_units "$temporary_directory" "$repository" "$service_user" "$service_group" "$service_home" \
        "$source_name" "$source_ref" "$script_dir"
    sudo install -m 0644 "$temporary_directory/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
    sudo install -m 0644 "$temporary_directory/$TIMER_NAME" "/etc/systemd/system/$TIMER_NAME"
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$TIMER_NAME"
    log "installed daily timer for $service_user using controller $repository"
    log "start one full run now with: sudo systemctl start --no-block $SERVICE_NAME"
}

main "$@"
