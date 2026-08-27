# Headless Locust benchmark: throughput (RPS) plus P50 / P95 / P99 latency.
#
# Prerequisites: API running (`python src/api/main.py`) and locust installed.
#
# Usage (from repo root):
#   powershell -File src/benchmarks/run_benchmark.ps1
#   $env:USERS=50; $env:SPAWN_RATE=10; $env:RUN_TIME="2m"; powershell -File src/benchmarks/run_benchmark.ps1

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$HostUrl = if ($env:HOST) { $env:HOST } else { "http://127.0.0.1:8000" }
$Users = if ($env:USERS) { $env:USERS } else { "20" }
$SpawnRate = if ($env:SPAWN_RATE) { $env:SPAWN_RATE } else { "5" }
$RunTime = if ($env:RUN_TIME) { $env:RUN_TIME } else { "1m" }
$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { Join-Path $Root "src\benchmarks\results" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Running headless Locust against $HostUrl"
Write-Host "  users=$Users spawn-rate=$SpawnRate run-time=$RunTime"
Write-Host "  CSV prefix: $OutDir\locust"
Write-Host "  HTML report: $OutDir\report.html"
Write-Host "  Metrics: Requests/s (throughput), 50%/95%/99% columns (P50/P95/P99)"

locust `
  -f (Join-Path $Root "src\benchmarks\locustfile.py") `
  --headless `
  --host $HostUrl `
  --users $Users `
  --spawn-rate $SpawnRate `
  --run-time $RunTime `
  --csv (Join-Path $OutDir "locust") `
  --html (Join-Path $OutDir "report.html")

Write-Host ""
Write-Host "Done. Open $OutDir\locust_stats.csv for P50/P95/P99 and RPS,"
Write-Host "or $OutDir\report.html for the full summary."
