#!/usr/bin/env bash
# Headless Locust benchmark: throughput (RPS) plus P50 / P95 / P99 latency.
#
# Prerequisites: API running (python src/api/main.py) and locust installed.
#
# Usage (from repo root):
#   bash src/benchmarks/run_benchmark.sh
#   USERS=50 SPAWN_RATE=10 RUN_TIME=2m HOST=http://127.0.0.1:8000 bash src/benchmarks/run_benchmark.sh
#
# PowerShell equivalent:
#   locust -f src/benchmarks/locustfile.py --headless --host http://127.0.0.1:8000 `
#     --users 20 --spawn-rate 5 --run-time 1m `
#     --csv src/benchmarks/results/locust --html src/benchmarks/results/report.html

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${HOST:-http://127.0.0.1:8000}"
USERS="${USERS:-20}"
SPAWN_RATE="${SPAWN_RATE:-5}"
RUN_TIME="${RUN_TIME:-1m}"
OUT_DIR="${OUT_DIR:-$ROOT/src/benchmarks/results}"

mkdir -p "$OUT_DIR"

echo "Running headless Locust against ${HOST}"
echo "  users=${USERS} spawn-rate=${SPAWN_RATE} run-time=${RUN_TIME}"
echo "  CSV prefix: ${OUT_DIR}/locust"
echo "  HTML report: ${OUT_DIR}/report.html"
echo "  Metrics: Requests/s (throughput), 50%/95%/99% columns (P50/P95/P99)"

locust \
  -f "$ROOT/src/benchmarks/locustfile.py" \
  --headless \
  --host "$HOST" \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$RUN_TIME" \
  --csv "$OUT_DIR/locust" \
  --html "$OUT_DIR/report.html"

echo
echo "Done. Open ${OUT_DIR}/locust_stats.csv for P50/P95/P99 and RPS,"
echo "or ${OUT_DIR}/report.html for the full summary."
