#!/usr/bin/env bash
set -euo pipefail
config=${1:?usage: launch_local_service_with_vectordb.sh CONFIG [SERVICE_PORT]}
service_port=${2:-7670}
retriever service start --config "$config" --port "$service_port" --launch-vectordb &
service_pid=$!
cleanup() { kill "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 150); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Retriever Service exited before readiness checks completed." >&2
    wait "$service_pid"
  fi
  if curl --fail --silent --show-error "http://127.0.0.1:7671/v1/health" >/dev/null \
    && curl --fail --silent --show-error "http://127.0.0.1:${service_port}/v1/health" >/dev/null; then
    echo "Retriever Service and local VectorDB are healthy and ready for requests."
    wait "$service_pid"
    exit 0
  fi
  sleep 0.2
done
echo "Timed out waiting for Retriever Service and VectorDB readiness." >&2
exit 1
