$ErrorActionPreference = 'Stop'
$SolverRoot = $PSScriptRoot
$DashboardUrl = 'http://127.0.0.1:8766/'
$DashboardStateUrl = $DashboardUrl + 'api/state'

function ConvertTo-DiamondInstallationPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    # Full Windows drive or UNC paths only; never resolve an untrusted relative
    # identity against this shell's working directory.
    if ($Path -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+(?:[\\/]|$))') {
        throw 'The dashboard returned an invalid installation path.'
    }
    return [System.IO.Path]::GetFullPath($Path).Replace('/', '\').TrimEnd([char]'\')
}

function Assert-DiamondDashboardInstallation {
    param($Response)
    if ($null -eq $Response -or $Response.service -cne 'diamond-dilemma' -or
        $Response.installation_root -isnot [string] -or
        [string]::IsNullOrWhiteSpace($Response.installation_root)) {
        throw 'Port 8766 has an unknown dashboard identity. No server was started. Identify or close the existing service before opening this checkout.'
    }
    $ExpectedRoot = ConvertTo-DiamondInstallationPath $SolverRoot
    $ExistingRoot = ConvertTo-DiamondInstallationPath $Response.installation_root
    if (-not [string]::Equals($ExpectedRoot, $ExistingRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Port 8766 belongs to another Diamond Dilemma installation: $ExistingRoot. This checkout is $ExpectedRoot. No server was started or stopped."
    }
}

function Test-DiamondDashboardPortListening {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Connection = $Client.BeginConnect('127.0.0.1', 8766, $null, $null)
        if (-not $Connection.AsyncWaitHandle.WaitOne(500)) {
            throw 'Port availability could not be verified within the probe timeout.'
        }
        $Client.EndConnect($Connection)
        return $true
    } catch {
        $Cause = $_.Exception
        while ($null -ne $Cause.InnerException) { $Cause = $Cause.InnerException }
        if ($Cause -is [System.Net.Sockets.SocketException] -and
            $Cause.SocketErrorCode -eq [System.Net.Sockets.SocketError]::ConnectionRefused) {
            return $false
        }
        throw 'Port 8766 availability could not be verified. No competing dashboard will be started.'
    } finally {
        $Client.Dispose()
    }
}

function Get-DiamondDashboardState {
    param([int]$TimeoutSec = 2)
    try {
        $Response = Invoke-RestMethod -Uri $DashboardStateUrl -TimeoutSec $TimeoutSec
    } catch {
        if (Test-DiamondDashboardPortListening) {
            throw 'Port 8766 is occupied, but its dashboard installation could not be verified. No competing dashboard will be started.'
        }
        return $null
    }
    # Keep identity rejection outside the network catch: it must never turn into
    # the absent-server path and cause a second process to be launched.
    Assert-DiamondDashboardInstallation $Response
    return $Response
}

function Open-DiamondDashboard {
    $DashboardReady = $null -ne (Get-DiamondDashboardState)
    if (-not $DashboardReady) {
        if (-not (Test-Path -LiteralPath (Join-Path $SolverRoot 'data\tiles.json'))) {
            throw 'Puzzle data is missing. Run Setup.cmd to prepare it before opening the dashboard.'
        }
        $SolverPython = Join-Path $SolverRoot 'venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $SolverPython)) { throw 'Run Setup.ps1 first.' }
        $ServerScript = Join-Path $SolverRoot 'dashboard_server.py'
        Start-Process -FilePath $SolverPython -ArgumentList ('"' + $ServerScript + '"') -WorkingDirectory $SolverRoot -WindowStyle Hidden
        for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
            Start-Sleep -Milliseconds 200
            if ($null -ne (Get-DiamondDashboardState -TimeoutSec 1)) {
                $DashboardReady = $true
                break
            }
        }
    }
    if (-not $DashboardReady) { throw 'Dashboard could not start; check whether port 8766 is in use.' }
    Start-Process $DashboardUrl
}

# Dot-sourcing exposes the probe functions for isolated tests without opening
# sockets, starting a server, or opening a browser.
if ($MyInvocation.InvocationName -ne '.') { Open-DiamondDashboard }
