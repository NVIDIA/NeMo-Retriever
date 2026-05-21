#!/usr/bin/env bash
set -euo pipefail

install_ffmpeg="$(printf "%s" "${INSTALL_FFMPEG:-false}" | tr '[:upper:]' '[:lower:]')"

case "${install_ffmpeg}" in
    1|true|yes|on)
        if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
            echo "INSTALL_FFMPEG=${INSTALL_FFMPEG}; ffmpeg and ffprobe are already available."
        else
            echo "INSTALL_FFMPEG=${INSTALL_FFMPEG}; installing ffmpeg and ffprobe with apt-get."
            sudo /usr/local/sbin/retriever-install-ffmpeg
        fi
        ;;
esac

exec "$@"
