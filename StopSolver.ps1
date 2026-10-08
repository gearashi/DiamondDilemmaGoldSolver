$ErrorActionPreference = 'Stop'
$RuntimePath = Join-Path $PSScriptRoot 'runtime'
if (-not (Test-Path -LiteralPath $RuntimePath)) {
    Write-Host 'No solver runtime exists yet.'
    exit 0
}
Set-Content -LiteralPath (Join-Path $RuntimePath 'stop.request') -Value 'stop'
Write-Host 'Stop requested. Wait for the solver to finish its current work and save the checkpoint.'
