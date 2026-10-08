[CmdletBinding()]
param([string]$SourceDir)
$ErrorActionPreference = 'Stop'
if ($SourceDir) { $SourceDir = [System.IO.Path]::GetFullPath($SourceDir) }
Set-Location -LiteralPath $PSScriptRoot
try {
    $venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
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
    Write-Host 'Installing pinned dependencies from the Python package index (network required)...'
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and the error above.' }
    $dataArgs = @((Join-Path $PSScriptRoot 'prepare_data.py'))
    if ($SourceDir) {
        $dataArgs += @('--source-dir', $SourceDir)
        Write-Host 'Preparing data from your local source GIFs...'
    } else {
        Write-Host 'Verifying local data; missing source GIFs will be downloaded from jaapsch.net...'
    }
    & $venvPython @dataArgs
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed. Existing conflicting files were preserved; see the error above.' }
    Write-Host 'Setup complete. An NVIDIA GPU and a CUDA-12-compatible driver are required to run the solver.'
    exit 0
} catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 1
}
