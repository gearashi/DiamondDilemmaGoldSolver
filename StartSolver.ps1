param([double]$Hours = 1, [int]$Replicas = 131072, [int]$Seed = 20261007, [switch]$Fresh, [switch]$Unlimited)
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
$SolverArgs = @((Join-Path $PSScriptRoot 'systematic_search.py'), '--stop-file-initialized', '--seconds', [string]$SolverSeconds, '--replicas', [string]$Replicas, '--seed', [string]$Seed)
if (-not $Fresh) { $SolverArgs += '--resume' }
& $SolverPython @SolverArgs
exit $LASTEXITCODE
