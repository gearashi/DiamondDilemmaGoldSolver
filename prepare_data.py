"""Prepare local puzzle data from three pinned source diagrams.

No source images are distributed with this project. By default this command
fetches the three HTTPS URLs in source-data.json. --source-dir uses existing
GIFs offline. Existing conflicting files are rejected, never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE_PAGE = 'https://www.jaapsch.net/puzzles/diamdil.htm'
SOURCE_NAMES = ('tilessilver.gif', 'tilesred.gif', 'tilesblue.gif')
SOURCE_URLS = {name: 'https://www.jaapsch.net/puzzles/images/diamdil/' + name
               for name in SOURCE_NAMES}
GENERATED = ('tiles.json', 'endpoint_audit.json', 'extraction_audit.json',
             'extraction_silver.png', 'extraction_red.png', 'extraction_blue.png')
MAX_SOURCE_BYTES = 2 * 1024 * 1024
SOCKET_TIMEOUT = 20
DOWNLOAD_DEADLINE = 60


class PreparationError(ValueError):
    """An input or generated artifact did not meet the pinned requirements."""


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise PreparationError(f'Cannot read JSON artifact {path}: {exc}') from exc


def load_manifest(path=ROOT / 'source-data.json'):
    manifest = read_json(path)
    sources = manifest.get('sources', [])
    if len(sources) != 3 or {x.get('name') for x in sources} != set(SOURCE_NAMES):
        raise PreparationError('Source manifest must contain exactly the three named GIFs.')
    for source in sources:
        if source.get('url') != SOURCE_URLS[source['name']]:
            raise PreparationError('Source URL is outside the three-URL allowlist.')
        digest = source.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise PreparationError('Source manifest contains an invalid SHA-256 digest.')
    expected = manifest.get('expected', {})
    if (expected.get('tile_count'), expected.get('segment_count'),
            expected.get('endpoint_count')) != (160, 365, 730):
        raise PreparationError('Unexpected puzzle dimensions in source manifest.')
    if len(expected.get('topology_sha256', '')) != 64:
        raise PreparationError('Source manifest lacks the expected topology fingerprint.')
    return manifest


def protected_child(directory, name):
    """Only direct regular files in the selected output directory are permitted."""
    directory = Path(directory).resolve()
    if Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
        raise PreparationError(f'Unsafe artifact filename: {name}')
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory:
        raise PreparationError(f'Artifact must not redirect outside its directory: {path}')
    if path.exists() and not path.is_file():
        raise PreparationError(f'Artifact path is not a regular file: {path}')
    return path


def verify_source(path, source):
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise PreparationError(f'Missing, empty, or oversized source image: {path}')
    if sha256(path) != source['sha256']:
        raise PreparationError(f'SHA-256 mismatch for {path}. Existing files were not overwritten; '
                               'move the conflicting file aside after inspecting it.')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise PreparationError(f'Source redirected to {newurl}; automatic redirects are disabled.')


def download_source(source, destination):
    if source['name'] not in SOURCE_URLS or source['url'] != SOURCE_URLS[source['name']]:
        raise PreparationError('Download URL is outside the three-URL allowlist.')
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(source['url'], headers={'User-Agent': 'DiamondDilemmaGoldSolver/1'})
    deadline = time.monotonic() + DOWNLOAD_DEADLINE
    total = 0
    with opener.open(request, timeout=SOCKET_TIMEOUT) as response, destination.open('xb') as output:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) > MAX_SOURCE_BYTES:
            raise PreparationError('Source download exceeds the size limit.')
        while True:
            if time.monotonic() >= deadline:
                raise PreparationError('Source download exceeded its time limit.')
            chunk = response.read(min(65536, MAX_SOURCE_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SOURCE_BYTES:
                raise PreparationError('Source download exceeds the size limit.')
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    verify_source(destination, source)


def install_missing(staged, target):
    """Atomically create a destination without replacing any existing file."""
    target = protected_child(target.parent, target.name)
    try:
        os.link(staged, target)
    except FileExistsError:
        if sha256(staged) != sha256(target):
            raise PreparationError(f'Conflicting existing artifact: {target}')
    except OSError as exc:
        raise PreparationError(f'Cannot atomically install {target}: {exc}') from exc


def canonical_topology(data):
    tiles = data.get('tiles', [])
    if len(tiles) != 160:
        raise PreparationError('Expected exactly 160 tiles.')
    result = []
    for number, tile in enumerate(tiles, 1):
        group, prefix = ('silver', 'S') if number <= 32 else ('red', 'R') if number <= 80 else ('blue', 'B')
        if (tile.get('number'), tile.get('id'), tile.get('group')) != (number, f'{prefix}{number:02d}', group):
            raise PreparationError(f'Unexpected tile identity or ordering at tile {number}.')
        segments, used = [], set()
        for segment in tile.get('segments', []):
            if not isinstance(segment, list) or len(segment) != 2:
                raise PreparationError(f'Malformed segment on tile {number}.')
            points = []
            for point in segment:
                if (not isinstance(point, list) or len(point) != 2 or
                    any(type(x) is not int for x in point) or
                    not 0 <= point[0] <= 2 or not 1 <= point[1] <= 11):
                    raise PreparationError(f'Invalid endpoint on tile {number}.')
                key = tuple(point)
                if key in used:
                    raise PreparationError(f'Repeated endpoint on tile {number}.')
                used.add(key)
                points.append(key)
            segments.append(sorted(points))
        if not segments:
            raise PreparationError(f'Tile {number} has no segments.')
        result.append({'id': tile['id'], 'number': number, 'group': group,
                       'segments': sorted(segments)})
    return result


def topology_digest(data):
    payload = json.dumps(canonical_topology(data), separators=(',', ':'), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_artifacts(directory, manifest, *, require_all=True):
    expected_hashes = {s['name']: s['sha256'] for s in manifest['sources']}
    documents = {}
    for name in GENERATED:
        path = protected_child(directory, name)
        if not path.exists():
            if require_all:
                raise PreparationError(f'Missing generated artifact: {path}')
            continue
        if name.endswith('.json'):
            document = read_json(path)
            if document.get('source_sha256') != expected_hashes:
                raise PreparationError(f'Source provenance mismatch in {path}')
            documents[name] = document
        elif path.stat().st_size < 8 or path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise PreparationError(f'Invalid overlay PNG: {path}')
    tiles = documents.get('tiles.json')
    if tiles is not None:
        if topology_digest(tiles) != manifest['expected']['topology_sha256']:
            raise PreparationError('Tile topology differs from the independently audited release input.')
        if sum(len(t['segments']) for t in tiles['tiles']) != 365:
            raise PreparationError('Expected 365 gold segments.')
    endpoints = documents.get('endpoint_audit.json')
    if endpoints is not None and tiles is not None:
        found = {t['id']: sorted(map(tuple, t['endpoints'])) for t in endpoints.get('tiles', [])}
        wanted = {t['id']: sorted(tuple(p) for s in t['segments'] for p in s) for t in tiles['tiles']}
        if len(endpoints.get('tiles', [])) != 160 or found != wanted:
            raise PreparationError('Endpoint audit does not match tile topology.')
    audit = documents.get('extraction_audit.json')
    if audit is not None:
        if (audit.get('tile_count'), audit.get('gold_segment_count'), audit.get('gold_endpoint_count'),
                audit.get('unexplained_strong_gold_pixels')) != (160, 365, 730, 0):
            raise PreparationError('Extraction audit counts are inconsistent.')
        if audit.get('minimum_pairing_margin', 0) <= 0:
            raise PreparationError('Extraction audit has ambiguous pairings.')
    return all(protected_child(directory, name).exists() for name in GENERATED)


def prepare(data_dir=ROOT / 'data', source_dir=None):
    manifest = load_manifest()
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        source_dir = Path(source_dir).resolve()
        if not source_dir.is_dir():
            raise PreparationError(f'Offline source directory does not exist: {source_dir}')
    for name in (*SOURCE_NAMES, *GENERATED, 'preparation.json'):
        protected_child(data_dir, name)
    # Fail before downloading or extracting if existing local data conflicts.
    for source in manifest['sources']:
        target = protected_child(data_dir, source['name'])
        if target.exists():
            verify_source(target, source)
    complete = validate_artifacts(data_dir, manifest, require_all=False)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=data_dir) as temp:
        stage = Path(temp)
        for source in manifest['sources']:
            target = protected_child(data_dir, source['name'])
            staged = stage / source['name']
            if target.exists():
                shutil.copyfile(target, staged)
            elif source_dir is not None:
                origin = protected_child(source_dir, source['name'])
                verify_source(origin, source)
                shutil.copyfile(origin, staged)
            else:
                print(f'Downloading {source["url"]}', flush=True)
                download_source(source, staged)
            verify_source(staged, source)
            if not target.exists():
                install_missing(staged, target)
        if not complete:
            for script in ('audit_endpoints.py', 'extract_tiles.py'):
                path = ROOT / script
                if not path.is_file():
                    raise PreparationError(f'Missing extraction program: {path}')
                print(f'Running {script}', flush=True)
                subprocess.run([sys.executable, str(path), '--data-dir', str(stage)], check=True, cwd=ROOT)
            validate_artifacts(stage, manifest)
            for name in GENERATED:
                # Existing valid artifacts remain untouched during partial repair.
                target = protected_child(data_dir, name)
                if not target.exists():
                    install_missing(stage / name, target)
        validate_artifacts(data_dir, manifest)
        provenance = protected_child(data_dir, 'preparation.json')
        if not provenance.exists():
            record = {'schema_version': 1, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                      'source_page': SOURCE_PAGE, 'source_mode': 'offline' if source_dir else 'download-or-existing',
                      'source_sha256': {s['name']: s['sha256'] for s in manifest['sources']},
                      'topology_sha256': manifest['expected']['topology_sha256'],
                      'note': 'Verified against source diagrams; no physical tile comparison or solution claim.'}
            staged = stage / 'preparation.json'
            staged.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            install_missing(staged, provenance)
    print(f'Local data verified: 160 tiles, 365 segments, 730 endpoints ({data_dir})', flush=True)
    return data_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Offline directory containing the three original GIFs')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='Local output directory (default: ./data)')
    args = parser.parse_args(argv)
    try:
        prepare(args.data_dir, args.source_dir)
    except (PreparationError, OSError, subprocess.CalledProcessError, urllib.error.URLError) as exc:
        parser.exit(1, f'Data preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
