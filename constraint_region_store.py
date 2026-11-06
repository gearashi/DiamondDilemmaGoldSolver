"""Versioned, bounded JSON storage for completed regional reasoning.

Only locally generated complete tables are written.  Identity includes puzzle
fingerprint, target, exact regional and adjacent domains, and schema version.
Checksums detect torn/corrupted files.  This is a local computation cache, not a
format for accepting untrusted externally supplied mathematical certificates.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from io_utils import atomic_json, read_json_shared

VERSION = 'diamond-regional-table-1'


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value,list) else value


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


class RegionalTableStore:
    def __init__(self, directory, owner, max_bytes=64*1024*1024, max_entries=128):
        self.directory = Path(directory)
        self.owner = owner
        self.max_bytes, self.max_entries = max_bytes,max_entries
        self.hits = self.writes = self.rejections = 0
        self.last_error = None
        source = Path(__file__).resolve().parent
        names = ('constraint_region_store.py','constraint_regions.py','constraint_separator.py',
                 'constraint_pruning.py','validator.py','search_problem.py','geometry.py')
        self.producer = tuple((name,hashlib.sha256((source/name).read_bytes()).hexdigest()) for name in names)

    def _identity(self, key):
        return {'version':VERSION,'problem':self.owner.problem.fingerprint,
                'target':self.owner.target,'producer':self.producer,'key':key}

    def _path(self, key):
        digest = hashlib.sha256(_encoded(self._identity(key))).hexdigest()
        return self.directory / (digest+'.json')

    def load(self, key, check):
        check(); path = self._path(key)
        try:
            if not path.exists():
                return None
            if path.stat().st_size > self.max_bytes:
                raise ValueError('Regional cache entry exceeds byte budget.')
            record = read_json_shared(path); payload = record['payload']
            if record['sha256'] != hashlib.sha256(_encoded(payload)).hexdigest():
                raise ValueError('Regional cache checksum mismatch.')
            if _encoded(payload['identity']) != _encoded(self._identity(key)):
                raise ValueError('Regional cache identity mismatch.')
            cells,domains,_ = key
            table = {}
            for raw_signature,raw_support,count in payload['rows']:
                check(); signature = _tuple(raw_signature); support = tuple(raw_support)
                if len(signature) != 4 or len(support) != len(cells) or type(count) is not int or count < 1:
                    raise ValueError('Malformed regional table payload.')
                used,masks,pairing,cycles = signature
                if type(used) is not int or used < 0 or used.bit_count() != len(cells) or used >> self.owner.problem.n:
                    raise ValueError('Malformed regional inventory.')
                if len(masks) != len(self.owner._frontier(set(cells))) or any(type(v) is not int or not 0 <= v < 2048 for v in masks):
                    raise ValueError('Malformed regional masks.')
                if any(type(v) is not int or v <= 0 or v & ~d for v,d in zip(support,domains)):
                    raise ValueError('Malformed regional placement supports.')
                if type(cycles) is not int or cycles < 0 or any(len(p) != 2 or any(type(v) is not int or not 0 <= v < 33*self.owner.problem.n for v in p) for p in pairing):
                    raise ValueError('Malformed regional pairing.')
                if signature in table:
                    raise ValueError('Duplicate regional signature.')
                table[signature] = support,count
            if len(table) > self.owner.max_cached_summaries:
                raise ValueError('Regional cache exceeds summary budget.')
            supports = [0]*len(cells)
            for support,_ in table.values():
                supports = [a|b for a,b in zip(supports,support)]
            self.hits += 1
            return {'supports':tuple(supports),'completions':sum(c for _,c in table.values()),
                'summary_count':len(table),'summaries':tuple(sorted(table)), 'table':table,
                'frontier_edges':self.owner._frontier(set(cells)), 'decomposition':(),
                'transitions':0,'states':0,'merges':0,'joins':0,'peak_states':0,
                'peak_frontier_edges':0,'complete':True,'outside_pairing_assumptions':False,
                'persistent_cache':True}
        except (OSError,ValueError,KeyError,TypeError,OverflowError):
            self.rejections += 1
            return None

    def save(self, key, result, check):
        if not result.get('complete') or self.max_entries <= 0:
            return False
        check()
        payload = {'identity':self._identity(key),'rows':[
            (signature,support,count) for signature,(support,count) in sorted(result['table'].items())]}
        record = {'payload':payload,'sha256':hashlib.sha256(_encoded(payload)).hexdigest()}
        if len(_encoded(record)) > self.max_bytes:
            return False
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            # Eviction is intentionally not destructive: once the bounded store
            # is full, new entries simply remain in the in-memory cache.
            path = self._path(key)
            if not path.exists() and sum(1 for _ in self.directory.glob('*.json')) >= self.max_entries:
                return False
            atomic_json(path,record); self.writes += 1
            return True
        except OSError as error:
            self.last_error = str(error)
            return False
