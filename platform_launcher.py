"""Portable local dashboard and search launchers for Linux, macOS, and Windows."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from gpu_backends import choose_backend
from io_utils import read_json_shared

ROOT = Path(__file__).resolve().parent

def venv_python():
    return ROOT / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def validate_installation(payload):
    if not isinstance(payload, dict) or payload.get('service') != 'diamond-dilemma':
        raise RuntimeError('The port belongs to an unknown service. No dashboard was started or stopped.')
    identity = payload.get('installation_root')
    if not isinstance(identity, str) or not Path(identity).is_absolute():
        raise RuntimeError('The existing dashboard has no verifiable installation identity.')
    if Path(identity).resolve() != ROOT.resolve():
        raise RuntimeError(f'This port belongs to another installation: {identity}. Use its dashboard or a different port.')
    return payload

def _port_listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        raise RuntimeError('Could not verify whether the dashboard port is free.') from exc

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise RuntimeError('The local dashboard redirected unexpectedly; refusing to start another server.')

def read_dashboard(port=8766):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Port must be an integer between 1 and 65535.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(f'http://127.0.0.1:{port}/api/state', timeout=2) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise RuntimeError('The dashboard response is unexpectedly large.')
        payload = json.loads(body)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        if _port_listening(port):
            raise RuntimeError('The port is occupied but its dashboard identity could not be verified.') from exc
        return None
    return validate_installation(payload)

def require_setup():
    if not venv_python().is_file():
        raise RuntimeError('Run Setup.sh (Linux/macOS) or Setup.cmd (Windows) first.')
    if not (ROOT / 'data/tiles.json').is_file():
        raise RuntimeError('Puzzle data is missing. Run setup again without --skip-data.')

def open_dashboard(port=8766, no_browser=False):
    url = f'http://127.0.0.1:{port}/'
    if read_dashboard(port) is None:
        require_setup()
        runtime = ROOT / 'runtime'
        runtime.mkdir(exist_ok=True)
        command = [str(venv_python()), str(ROOT / 'dashboard_server.py'), '--port', str(port)]
        kwargs = {'cwd': str(ROOT), 'stdin': subprocess.DEVNULL}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        with (runtime / 'dashboard.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if read_dashboard(port) is not None:
                break
            if child.poll() is not None:
                raise RuntimeError('Dashboard exited before becoming ready. Check runtime/dashboard.log.')
            time.sleep(.2)
        else:
            raise RuntimeError('Dashboard startup timed out. Check runtime/dashboard.log.')
    print(url, flush=True)
    if not no_browser:
        if not webbrowser.open(url):
            print('Open the address above in your browser.', flush=True)
    return url

def solver_command(hours=1, unlimited=False, replicas=4096, seed=20261007, backend='auto', fresh=False):
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or not math.isfinite(hours) or hours < 0:
        raise ValueError('Hours must be finite and nonnegative.')
    if type(replicas) is not int or not 128 <= replicas <= 131072:
        raise ValueError('Choose between 128 and 131072 replicas.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative 32-bit integer.')
    choose_backend(backend)
    command = [str(venv_python()), str(ROOT / 'systematic_search.py'),
               '--seconds', str(0 if unlimited else hours * 3600),
               '--replicas', str(replicas), '--seed', str(seed), '--backend', backend]
    if not fresh:
        command.append('--resume')
    return command

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    dashboard = sub.add_parser('dashboard', help='Start the local dashboard')
    dashboard.add_argument('--port', type=int, default=8766)
    dashboard.add_argument('--no-browser', action='store_true')
    start = sub.add_parser('start', help='Run systematic search in this terminal')
    start.add_argument('--hours', type=float, default=1)
    start.add_argument('--unlimited', action='store_true')
    start.add_argument('--replicas', type=int, default=4096)
    start.add_argument('--seed', type=int, default=20261007)
    start.add_argument('--backend', choices=('auto', 'cuda', 'webgpu'), default='auto')
    start.add_argument('--fresh', action='store_true')
    sub.add_parser('stop', help='Request a graceful stop and checkpoint')
    sub.add_parser('status', help='Print saved search status')
    args = parser.parse_args(argv)
    try:
        if args.action == 'dashboard':
            open_dashboard(args.port, args.no_browser)
        elif args.action == 'start':
            command = solver_command(args.hours, args.unlimited, args.replicas, args.seed, args.backend, args.fresh)
            require_setup()
            return subprocess.call(command, cwd=ROOT)
        elif args.action == 'stop':
            runtime = ROOT / 'runtime'
            if runtime.exists():
                (runtime / 'stop.request').write_text('stop from launcher', encoding='utf-8')
                print('Stop requested. Wait for Stopped after the final checkpoint is saved.')
            else:
                print('No solver runtime exists yet.')
        else:
            path = ROOT / 'runtime/status.json'
            print(json.dumps(read_json_shared(path), indent=2) if path.exists() else 'No saved solver status yet.')
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
