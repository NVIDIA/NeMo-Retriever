#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "retriever-install-ffmpeg does not accept arguments." >&2
    exit 64
fi

export DEBIAN_FRONTEND=noninteractive
/usr/bin/apt-get update
/usr/bin/apt-get install -y --no-install-recommends ffmpeg
/usr/bin/apt-get clean
