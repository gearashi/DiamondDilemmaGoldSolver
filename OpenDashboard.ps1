[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8766,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$SolverRoot = $PSScriptRoot

function Invoke-DiamondLauncher {
    param([string]$Python, [string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Dashboard startup failed. See the error above and runtime/dashboard.log. For a Windows-reserved port, try OpenDashboard.cmd -Port 8767.'
    }
}

function Open-DiamondDashboard {
    # Shared launcher bypasses HTTP proxies, logs errors, verifies installation
    # identity and waits for our child between socket bind and first HTTP reply.
    $SolverPython = Join-Path $SolverRoot 'venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $SolverPython)) { throw 'Run Setup.cmd first.' }
    $LauncherArgs = @((Join-Path $SolverRoot 'platform_launcher.py'), 'dashboard', '--port', [string]$Port)
    if ($NoBrowser) { $LauncherArgs += '--no-browser' }
    Invoke-DiamondLauncher -Python $SolverPython -Arguments $LauncherArgs
}

if ($MyInvocation.InvocationName -ne '.') { Open-DiamondDashboard }
