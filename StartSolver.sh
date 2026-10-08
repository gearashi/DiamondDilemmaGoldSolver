#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python="$script_dir/venv/bin/python"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'Run Setup.sh first to create the local Python environment.' >&2
    exit 1
fi
exec "$python" "$script_dir/platform_launcher.py" start "$@"
