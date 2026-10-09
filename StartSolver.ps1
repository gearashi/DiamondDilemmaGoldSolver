param([double]$Hours = 1, [int]$Replicas = 131072, [int]$Seed = 20261007, [switch]$Fresh, [switch]$Unlimited, [ValidateSet('auto','cuda','webgpu')][string]$Backend = 'auto', [ValidateSet('gpu-dfs','dfs','cp','cp-sat','sat','hybrid')][string]$Algorithm = 'gpu-dfs')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$SolverPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $SolverPython)) { throw 'Run Setup.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'data\tiles.json'))) { throw 'Puzzle data is missing. Run Setup.cmd to prepare it before starting a search.' }
$SolverLock = Join-Path $PSScriptRoot 'runtime\run.lock'
if (Test-Path -LiteralPath $SolverLock) { throw 'A solver lock exists. Stop the active solver and wait for its checkpoint before starting another run.' }
$SolverStopFile = Join-Path $PSScriptRoot 'runtime\stop.request'
Remove-Item -LiteralPath $SolverStopFile -ErrorAction SilentlyContinue
$SolverSeconds = if ($Unlimited) { 0 } else { $Hours*3600 }
$SolverScript = if ($Algorithm -eq 'gpu-dfs') { 'systematic_search.py' } else { 'constraint_search.py' }
$SolverArgs = @((Join-Path $PSScriptRoot $SolverScript), '--stop-file-initialized', '--seconds', [string]$SolverSeconds, '--replicas', [string]$Replicas, '--seed', [string]$Seed, '--backend', $Backend)
if ($Algorithm -ne 'gpu-dfs') { $SolverArgs += @('--algorithm', $Algorithm) }
if (-not $Fresh) { $SolverArgs += '--resume' }
& $SolverPython @SolverArgs
exit $LASTEXITCODE
