[CmdletBinding()]
param(
    [string]$SourceDir,
    [ValidateSet('auto', 'cuda', 'webgpu')][string]$Backend = 'auto',
    [switch]$SkipData
)
$ErrorActionPreference = 'Stop'
if ($SourceDir) { $SourceDir = [System.IO.Path]::GetFullPath($SourceDir) }
Set-Location -LiteralPath $PSScriptRoot
try {
    $venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'venv')) { throw 'venv exists without a usable Windows Scripts/python.exe. Use a separate Windows checkout or move that environment aside before setup.' }
        $launcher = $null
        $launcherArgs = @()
        foreach ($candidate in @('py', 'python')) {
            $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $command) { continue }
            $candidateArgs = @()
            if ($candidate -eq 'py') { $candidateArgs = @('-3.12') }
            try {
                & $command.Source @candidateArgs -c 'import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8 else 1)' 2>$null
            } catch { continue }
            if ($LASTEXITCODE -eq 0) {
                $launcher = $command.Source
                $launcherArgs = $candidateArgs
                break
            }
        }
        if (-not $launcher) { throw 'Install 64-bit Python 3.12 from python.org with the Python launcher or add it to PATH, then run Setup.cmd again.' }
        & $launcher @launcherArgs -m venv (Join-Path $PSScriptRoot 'venv')
        if ($LASTEXITCODE -ne 0) { throw 'Python could not create the local virtual environment.' }
    }
    & $venvPython -c 'import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(chr(80))==8 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The existing venv must use 64-bit Python 3.12. Move it aside before rebuilding it.' }
    & $venvPython -c 'import os,sys;sys.exit(0 if sys.prefix != sys.base_prefix and os.path.realpath(sys.prefix)==os.path.realpath(sys.argv[1]) else 1)' (Join-Path $PSScriptRoot 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv must be an isolated environment belonging to this checkout.' }
    $requirements = @()
    if ($Backend -eq 'cuda') {
        $requirements += 'requirements.txt'
    } else {
        $requirements += 'requirements-webgpu.txt'
        if ($Backend -eq 'auto') {
            $nvidia = Get-Command 'nvidia-smi' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($nvidia) {
                try {
                    $gpuNames = @(& $nvidia.Source --query-gpu=name --format=csv,noheader 2>$null)
                    if ($LASTEXITCODE -eq 0 -and ($gpuNames -join '').Trim()) { $requirements += 'requirements.txt' }
                } catch { Write-Host 'NVIDIA driver query did not succeed; installing WebGPU dependencies only.' }
            }
        }
    }
    $pipArgs = @('-m', 'pip', '--require-virtualenv', 'install', '--no-user', '--disable-pip-version-check')
    foreach ($requirement in $requirements) { $pipArgs += @('-r', (Join-Path $PSScriptRoot $requirement)) }
    Write-Host ('Installing pinned dependencies for ' + $Backend + ': ' + ($requirements -join ', '))
    $savedPipEnvironment = @{}
    try {
        foreach ($name in @('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_ROOT', 'PIP_CONFIG_FILE')) {
            $savedPipEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        [Environment]::SetEnvironmentVariable('PIP_CONFIG_FILE', 'NUL', 'Process')
        & $venvPython @pipArgs
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and the error above.' }
    } finally {
        foreach ($name in $savedPipEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $savedPipEnvironment[$name], 'Process')
        }
    }
    if ($SkipData) {
        Write-Host 'Dependency setup complete; puzzle data preparation was skipped. Run prepare_data.py before searching.'
        exit 0
    }
    $dataArgs = @((Join-Path $PSScriptRoot 'prepare_data.py'))
    if ($SourceDir) {
        $dataArgs += @('--source-dir', $SourceDir)
        Write-Host 'Preparing data from your local source GIFs...'
    } else {
        Write-Host 'Verifying local data; missing source GIFs will be downloaded from jaapsch.net...'
    }
    & $venvPython @dataArgs
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed. Existing conflicting files were preserved; see the error above.' }
    Write-Host 'Setup complete. OpenDashboard.cmd starts the dashboard. GPU search requires a compatible GPU and driver; setup does not install drivers.'
    exit 0
} catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 1
}
