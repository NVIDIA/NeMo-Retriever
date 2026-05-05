#!/usr/bin/env bash
# Regenerate every uv.lock in the repo and fail if any changed.
# Run from the repo root.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CHANGED=0

while IFS= read -r lockfile; do
    dir="$(dirname "$lockfile")"
    echo "uv lock: $dir"
    (cd "$REPO_ROOT/$dir" && uv lock --quiet)
    if ! git -C "$REPO_ROOT" diff --quiet "$lockfile"; then
        echo "  ERROR: $lockfile is out of date — stage the regenerated file and re-commit."
        CHANGED=1
    fi
done < <(git -C "$REPO_ROOT" ls-files '*/uv.lock' 'uv.lock')

exit "$CHANGED"
