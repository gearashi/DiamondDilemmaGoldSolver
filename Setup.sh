#!/usr/bin/env bash
# Install into this checkout only; source diagrams remain optional local data.
set -Eeuo pipefail

fail() {
    printf 'Setup failed: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: bash Setup.sh [--backend auto|cuda|webgpu] [--python PATH] [--source-dir DIR] [--skip-data]

  --backend NAME    auto (default): WebGPU, plus CUDA if an NVIDIA query succeeds.
                    cuda: NVIDIA CUDA on Linux; webgpu: Metal/Vulkan/DX12.
  --python PATH     Use this 64-bit Python 3.12 interpreter to create venv/.
                    Otherwise try python3.12, then python3 on PATH.
  --source-dir DIR  Use the three existing source GIFs offline. Relative paths
                    are resolved from the directory where you invoke setup.
  --skip-data       Install dependencies only; do not prepare/download data.
  --help, -h        Show this help.

Installs pinned requirements into this checkout's venv/bin/python. No sudo or
system Python changes are made. Package installation needs network access;
missing puzzle diagrams are downloaded unless --source-dir or --skip-data is
used. Existing incompatible environments are preserved and reported as errors.
USAGE
}

trap 'fail "Unexpected error at line ${LINENO}. See the preceding error."' ERR
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
caller_dir=$PWD
python_request=''
source_dir=''
skip_data=false
backend=auto
while (($#)); do
    case "$1" in
        --backend)
            (($# >= 2)) && [[ -n "$2" ]] || fail '--backend requires auto, cuda, or webgpu.'
            backend=$2
            case "$backend" in auto|cuda|webgpu) ;; *) fail 'Backend must be auto, cuda, or webgpu.' ;; esac
            shift 2
            ;;
        --python)
            (($# >= 2)) && [[ -n "$2" ]] || fail '--python requires an interpreter path.'
            python_request=$2
            shift 2
            ;;
        --source-dir)
            (($# >= 2)) && [[ -n "$2" ]] || fail '--source-dir requires a directory.'
            source_dir=$2
            shift 2
            ;;
        --skip-data) skip_data=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown argument: $1. Use --help for usage." ;;
    esac
done

platform=$(uname -s)
case "$platform" in Linux|Darwin) ;; *) fail 'Setup.sh supports Linux and macOS; use Setup.cmd on Windows.' ;; esac
if [[ "$platform" == Darwin && "$backend" == cuda ]]; then
    fail 'CUDA is not supported on macOS. Use --backend webgpu for native Metal GPU search.'
fi
requirements=()
if [[ "$backend" == cuda ]]; then
    requirements+=(requirements.txt)
else
    requirements+=(requirements-webgpu.txt)
    if [[ "$backend" == auto && "$platform" != Darwin ]] && command -v nvidia-smi >/dev/null 2>&1; then
        if gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) && [[ -n "${gpu_names//[[:space:]]/}" ]]; then
            requirements+=(requirements.txt)
        fi
    fi
fi
for requirement in requirements-common.txt "${requirements[@]}"; do
    [[ -f "$script_dir/$requirement" ]] || fail "Missing $requirement next to Setup.sh."
done
if [[ -n "$source_dir" ]]; then
    [[ "$source_dir" = /* ]] || source_dir="$caller_dir/$source_dir"
    [[ -d "$source_dir" ]] || fail "Source directory does not exist: $source_dir"
    source_dir=$(cd -- "$source_dir" && pwd -P)
fi
if [[ "$skip_data" == false ]]; then
    [[ -f "$script_dir/prepare_data.py" ]] || fail 'Missing prepare_data.py next to Setup.sh.'
fi

version_check='import struct,sys;sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize("P")==8 else 1)'
venv_dir="$script_dir/venv"
venv_python="$venv_dir/bin/python"
python_command=''
if [[ -n "$python_request" ]]; then
    if [[ "$python_request" == */* ]]; then
        [[ "$python_request" = /* ]] || python_request="$caller_dir/$python_request"
        [[ -x "$python_request" ]] || fail "Python interpreter is not executable: $python_request"
        python_command=$python_request
    else
        python_command=$(command -v -- "$python_request") || fail "Python interpreter was not found: $python_request"
    fi
    "$python_command" -c "$version_check" >/dev/null 2>&1 || fail 'The requested interpreter must be 64-bit Python 3.12.'
fi

if [[ ! -x "$venv_python" ]]; then
    if [[ -e "$venv_dir" || -L "$venv_dir" ]]; then
        fail 'venv/ exists without a usable POSIX bin/python. Use a separate checkout for this operating system or move that environment aside before setup.'
    fi
    if [[ -z "$python_command" ]]; then
        for candidate in python3.12 python3; do
            if candidate_path=$(command -v -- "$candidate") && "$candidate_path" -c "$version_check" >/dev/null 2>&1; then
                python_command=$candidate_path
                break
            fi
        done
    fi
    [[ -n "$python_command" ]] || fail '64-bit Python 3.12 was not found. Install it with venv support, or supply --python PATH.'
    printf 'Creating local Python environment: %s\n' "$venv_dir"
    "$python_command" -m venv "$venv_dir" || fail 'Could not create venv/. Ensure this Python has venv/ensurepip support and the checkout is writable.'
fi
[[ -x "$venv_python" ]] || fail 'Environment creation did not produce venv/bin/python.'
"$venv_python" -c "$version_check" >/dev/null 2>&1 || fail 'The existing venv/ must use 64-bit Python 3.12. Move it aside before rebuilding.'
# A misplaced symlink to system Python must never turn this into a global pip install.
"$venv_python" -c 'import os,sys;sys.exit(0 if sys.prefix != sys.base_prefix and os.path.realpath(sys.prefix)==os.path.realpath(sys.argv[1]) else 1)' "$venv_dir" \
    || fail 'venv/bin/python is not an isolated environment belonging to this checkout.'

printf 'Installing pinned dependencies for %s into venv/ (network required)...\n' "$backend"
pip_requirements=()
for requirement in "${requirements[@]}"; do
    pip_requirements+=(-r "$script_dir/$requirement")
done
if ! (
    unset PIP_TARGET PIP_PREFIX PIP_USER PIP_ROOT
    export PIP_CONFIG_FILE=/dev/null
    "$venv_python" -m pip --require-virtualenv install --no-user --disable-pip-version-check "${pip_requirements[@]}"
); then
    fail 'Dependency installation failed. Check network access, package availability, and the error above.'
fi

if [[ "$skip_data" == true ]]; then
    printf '%s\n' 'Dependency setup complete; puzzle data preparation was skipped.'
else
    data_args=("$script_dir/prepare_data.py")
    if [[ -n "$source_dir" ]]; then
        data_args+=(--source-dir "$source_dir")
        printf '%s\n' 'Preparing puzzle data from your local source GIFs...'
    else
        printf '%s\n' 'Verifying local data; missing source GIFs will be downloaded from jaapsch.net...'
    fi
    "$venv_python" "${data_args[@]}" || fail 'Data preparation failed. See the error above; conflicting existing files were preserved.'
    printf '%s\n' 'Setup complete. Use ./OpenDashboard.sh to open the dashboard.'
fi
printf '%s\n' 'GPU search requires a compatible GPU and driver. WebGPU uses native Metal on macOS and Vulkan/DX12 on supported PCs. Setup does not install drivers.'
