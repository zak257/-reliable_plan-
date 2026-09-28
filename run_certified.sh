#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_python="${CERTIFIED_PLAN_PYTHON:-${CAP_PLAN_PYTHON:-/home/yzk/cap_plan/venv/bin/python}}"
cd "$task_dir"
exec "$task_python" -m certified_reliability_planning "$@"
