#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python.module> [args ...]"
  echo "Example: $0 scripts.benchmarks.nftf.nftf_comparison --epochs 100"
  exit 1
fi

script_module="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${repo_root}/logs"
log_file="${log_dir}/log.out"

mkdir -p "${log_dir}"

echo "Launching: python -m ${script_module} $*"
echo "Logging to: ${log_file}"

# Write a small header so the file updates immediately.
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: python -u -m ${script_module} $*"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Working dir: $(pwd)"
} >> "${log_file}"

# Use unbuffered Python output so logs stream in real time.
nohup python -u -m "${script_module}" "$@" >> "${log_file}" 2>&1 &
pid=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID: ${pid}" >> "${log_file}"
echo "Started in background with PID ${pid}"
echo "Watch logs with: tail -f ${log_file}"