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
$versionCheck = 'import platform,struct,sys,sysconfig; v=sys.version_info[:2]; ok=platform.python_implementation()==''CPython'' and v in ((3,12),(3,15)) and struct.calcsize(''P'')==8 and not sysconfig.get_config_var(''Py_GIL_DISABLED''); print(str(v[0])+''.''+str(v[1])) if ok else None; sys.exit(0 if ok else 1)'
$setupDir = Join-Path $PSScriptRoot '.solver-setup'
$venv = Join-Path $PSScriptRoot 'venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

function Find-FullPython {
    # Python may be registered with the launcher, on PATH, or private to this kit.
    foreach ($candidate in @('py', 'python', (Join-Path $setupDir 'python312\tools\python.exe'))) {
        $command = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) { continue }
        $selector = @()
        if ($candidate -eq 'py') { $selector = @('-3.12') }
        try {
            $version = & $command.Source @selector -c $versionCheck 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -eq '3.12') {
                $exe = & $command.Source @selector -c 'import sys; print(sys._base_executable)'
                if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $exe -PathType Leaf)) { return $exe }
            }
        } catch { continue }
    }
}

function Install-PrivatePython {
    # The Python Software Foundation's x64 NuGet distribution includes venv and ensurepip.
    $url = 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg'
    $sha256 = '0EB85C2DFCCCCF1B17352DE4C397F69194035B7D37149EACC16F1147D93DE3B8'
    $id = [guid]::NewGuid().ToString('N')
    $download = Join-Path $setupDir "$id.nupkg"
    $unpacked = Join-Path $setupDir $id
    $destination = Join-Path $setupDir 'python312'
    Write-Host 'Downloading a private x64 Python 3.12 for this kit. Your system Python stays installed.'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        (New-Object Net.WebClient).DownloadFile($url, $download)
        $hasher = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($download)
        try { $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ($digest -ne $sha256) { throw 'Private Python download failed SHA-256 verification.' }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($download, $unpacked)
        if (Test-Path -LiteralPath $destination) {
            [IO.Directory]::Move($destination, (Join-Path $setupDir "previous-python312-$id"))
        }
        [IO.Directory]::Move($unpacked, $destination)
    } finally {
        if (Test-Path -LiteralPath $download) { Remove-Item -LiteralPath $download }
    }
    return (Join-Path $destination 'tools\python.exe')
}

$setupLock = $null
$backup = $null
try {
    $null = [IO.Directory]::CreateDirectory($setupDir)
    $setupLock = [IO.File]::Open((Join-Path $setupDir 'setup.lock'), 'OpenOrCreate', 'Write', 'None')
    $launcher = $null
    if ($Python) {
        $command = Get-Command $Python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) { throw "The requested Python executable was not found: $Python" }
        $version = & $command.Source -c $versionCheck
        if ($LASTEXITCODE -ne 0) { throw 'The requested interpreter must be regular 64-bit Python 3.12 or 3.15.' }
        if ($version -eq '3.12') { $launcher = $command.Source }
        else { Write-Host 'Python 3.15 detected. The pinned CuPy, OR-Tools and python-sat wheels require a side-by-side Python 3.12 for the full Windows profile.' }
    }
    $existingVersion = $null
    if (Test-Path -LiteralPath $venv) {
        if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) { throw 'venv exists without Scripts/python.exe. Use a separate Windows checkout or move the foreign environment aside.' }
        $existingVersion = & $venvPython -c $versionCheck
        if ($LASTEXITCODE -ne 0) { throw 'The existing venv must use regular 64-bit Python 3.12 or 3.15.' }
        & $venvPython -c 'import os,sys;sys.exit(0 if sys.prefix != sys.base_prefix and os.path.realpath(sys.prefix)==os.path.realpath(sys.argv[1]) else 1)' $venv
        if ($LASTEXITCODE -ne 0 -or ((Get-Item -LiteralPath $venv).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'venv must be an isolated environment belonging to this checkout.' }
    }
    if ($existingVersion -ne '3.12') {
        if (-not $launcher) { $launcher = Find-FullPython }
        if (-not $launcher) { $launcher = Install-PrivatePython }
        if ($existingVersion -eq '3.15') {
            $backup = Join-Path $setupDir ('previous-venv-' + [guid]::NewGuid().ToString('N'))
            [IO.Directory]::Move($venv, $backup)
            Write-Host "Upgrading the local 3.15 environment to the full 3.12 profile. Previous environment saved under $backup."
        }
        & $launcher -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Python could not create the local virtual environment.' }
    }
    $version = & $venvPython -c $versionCheck
    if ($LASTEXITCODE -ne 0 -or $version -ne '3.12') { throw 'The full Windows profile requires a regular x64 Python 3.12 venv.' }
    $requirements = @('requirements-webgpu.txt')
    if ($Backend -ne 'webgpu') { $requirements += 'requirements.txt' }
    $pipArgs = @('-m', 'pip', '--require-virtualenv', 'install', '--no-user', '--only-binary=:all:', '--disable-pip-version-check')
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
    & $venvPython (Join-Path $PSScriptRoot 'build_native.py') --install-windows-bundle
    if ($LASTEXITCODE -ne 0) { throw 'The Windows x64 native solver bundle could not be activated.' }
    & $venvPython (Join-Path $PSScriptRoot 'windows_profile.py') $Backend
    if ($LASTEXITCODE -ne 0) { throw 'Full Windows engine verification failed. Run .\Setup.cmd to repair this installation.' }
    if ($Backend -ne 'webgpu') {
        & $venvPython (Join-Path $PSScriptRoot 'cuda_probe.py') --record
        if ($LASTEXITCODE -eq 2 -and $Backend -eq 'auto') {
            Write-Warning 'CUDA hardware unavailable. Check for a missing NVIDIA display driver. CPU and WebGPU engines are installed.'
        } elseif ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed. See the probe result above.' }
    }
    # A data download failure can be retried without undoing a working software installation.
    $backup = $null
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
    if ($backup -and (Test-Path -LiteralPath $backup)) {
        if (Test-Path -LiteralPath $venv) {
            [IO.Directory]::Move($venv, (Join-Path $setupDir ('incomplete-venv-' + [guid]::NewGuid().ToString('N'))))
        }
        [IO.Directory]::Move($backup, $venv)
        Write-Host 'Restored the previous local environment after the failed upgrade.'
    }
    Write-Error -Message $_ -ErrorAction Continue
    exit 1
} finally {
    if ($setupLock) { $setupLock.Dispose() }
}
