#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_python="${CAP_PLAN_PYTHON:-/home/yzk/cap_plan/venv/bin/python}"
cd "$task_dir"
exec "$task_python" -m polar_reliability_planning.recovery_cli "$@"
