$ErrorActionPreference = 'Stop'
$StatusPath = Join-Path $PSScriptRoot 'runtime\status.json'
if (Test-Path -LiteralPath $StatusPath) {
    Get-Content -LiteralPath $StatusPath
} else {
    Write-Host 'No saved solver status yet. Open the dashboard to start a search.'
}
