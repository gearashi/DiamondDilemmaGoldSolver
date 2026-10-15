[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$Python,
    [ValidateSet('auto', 'cuda', 'webgpu')][string]$Backend = 'auto',
    [switch]$SkipData
)
$ErrorActionPreference = 'Stop'
if ($SourceDir) { $SourceDir = [System.IO.Path]::GetFullPath($SourceDir) }
if ($Python -and ($Python.Contains('\') -or $Python.Contains('/'))) { $Python = [System.IO.Path]::GetFullPath($Python) }
Set-Location -LiteralPath $PSScriptRoot
try {
    # Keep full engine support on 3.12; 3.15 currently has a DFS/WebGPU profile.
    # Free-threaded builds have different native package compatibility.
    $versionCheck = 'import struct,sys,sysconfig;v=str(sys.version_info.major)+chr(46)+str(sys.version_info.minor);ok=sys.version_info[:2] in ((3,12),(3,15)) and struct.calcsize(chr(80))==8 and not sysconfig.get_config_var(''Py_GIL_DISABLED'') and (len(sys.argv)<2 or v==sys.argv[1]);print(v) if ok else None;sys.exit(0 if ok else 1)'
    $requestedVersion = $null
    $launcher = $null
    $launcherArgs = @()
    if ($Python) {
        $command = Get-Command $Python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) { throw "The requested Python executable was not found: $Python" }
        $launcher = $command.Source
        $requestedVersion = & $launcher -c $versionCheck
        if ($LASTEXITCODE -ne 0) { throw 'The requested interpreter must be regular 64-bit Python 3.12 or 3.15. Free-threaded builds are not supported.' }
    }
    $venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'venv')) { throw 'venv exists without a usable Windows Scripts/python.exe. Use a separate Windows checkout or move that environment aside before setup.' }
        if (-not $launcher) {
            foreach ($wantedVersion in @('3.12', '3.15')) {
                foreach ($candidate in @('py', 'python')) {
                    $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
                    if (-not $command) { continue }
                    $candidateArgs = @()
                    if ($candidate -eq 'py') { $candidateArgs = @('-' + $wantedVersion) }
                    try {
                        $candidateVersion = & $command.Source @candidateArgs -c $versionCheck $wantedVersion 2>$null
                    } catch { continue }
                    if ($LASTEXITCODE -eq 0) {
                        $launcher = $command.Source
                        $launcherArgs = $candidateArgs
                        break
                    }
                }
                if ($launcher) { break }
            }
        }
        if (-not $launcher) { throw 'Install regular 64-bit Python 3.12 (all engines) or 3.15 (DFS/WebGPU), or supply -Python PATH. Free-threaded builds are not supported.' }
        & $launcher @launcherArgs -m venv (Join-Path $PSScriptRoot 'venv')
        if ($LASTEXITCODE -ne 0) { throw 'Python could not create the local virtual environment.' }
    }
    $venvVersion = & $venvPython -c $versionCheck
    if ($LASTEXITCODE -ne 0) { throw 'The existing venv must use regular 64-bit Python 3.12 or 3.15. Move it aside before rebuilding it. Free-threaded builds are not supported.' }
    if ($requestedVersion -and $requestedVersion -ne $venvVersion) { throw "The requested Python $requestedVersion does not match existing venv Python $venvVersion. The existing environment was preserved; use a separate checkout or move it aside before changing Python versions." }
    & $venvPython -c 'import os,sys;sys.exit(0 if sys.prefix != sys.base_prefix and os.path.realpath(sys.prefix)==os.path.realpath(sys.argv[1]) else 1)' (Join-Path $PSScriptRoot 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv must be an isolated environment belonging to this checkout.' }
    $requirements = @()
    if ($venvVersion -eq '3.15') {
        if ($Backend -eq 'cuda') { throw 'CUDA is unavailable in the Python 3.15 dependency profile. Use -Backend webgpu, or Python 3.12 for CUDA and CP/SAT engines.' }
        $requirements += 'requirements-py315.txt'
        Write-Host 'Python 3.15 profile: DFS and WebGPU search. CP, CP-SAT, SAT, and their optional assistance require Python 3.12.'
    } elseif ($Backend -eq 'cuda') {
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
