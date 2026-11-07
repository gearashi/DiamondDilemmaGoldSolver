"""Bounded application-owned reasoning cache; never an external proof format.

Every record is independently atomic and source-bound. A cancelled/failed write
leaves the older record usable. Concurrent writers may lose optional reuse but
cannot manufacture authority: readers verify identity, digest and component
structure. Eviction never changes pending search coverage.
"""
from __future__ import annotations
import hashlib,json,os,tempfile,time
from pathlib import Path

AUTHORITY='trusted_local_reasoning_not_external_proof'
VERSION=1
MAX_RECORD_BYTES=32*1024*1024
STAT_KEYS=('reasoning_cache_reads','reasoning_cache_hits','reasoning_cache_invalid',
           'reasoning_cache_writes','reasoning_cache_write_bytes','reasoning_cache_unchanged',
           'reasoning_cache_evictions','reasoning_cache_write_failures','reasoning_cache_global_evictions','reasoning_cache_stale_temporaries')


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def source_binding(paths):
    result={}
    for path in paths:
        path=Path(path).resolve()
        if path.name in result:raise ValueError('Duplicate local source name.')
        result[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class LocalReasoningStore:
    """An optional cache under a caller-owned directory, not a search frontier.

    The context must include puzzle, objective, original and effective roots.
    Policy-specific cache keys are supplied by each component. A component's
    source list invalidates only that component when implementation changes.
    Unknown versions and corrupted/evicted records always return a cold miss.
    """
    def __init__(self,directory,context,*,max_bytes=64*1024*1024,max_entries=128,global_max_bytes=512*1024*1024,global_max_entries=4096):
        if type(max_bytes)is not int or max_bytes<0 or type(max_entries)is not int or max_entries<0:
            raise ValueError('Invalid reasoning cache bounds.')
        if type(global_max_bytes)is not int or global_max_bytes<0 or type(global_max_entries)is not int or global_max_entries<0:raise ValueError('Invalid global reasoning cache bounds.')
        self.global_max_bytes=global_max_bytes;self.global_max_entries=global_max_entries
        body=encoded(context)
        if len(body)>1024*1024:raise ValueError('Reasoning context exceeds bound.')
        self.context=json.loads(body);self.context_hash=hashlib.sha256(body).hexdigest()
        self.directory=Path(directory).resolve()/'local-reasoning-v1'/self.context_hash
        self.max_bytes=max_bytes;self.max_entries=max_entries;self.stats=dict.fromkeys(STAT_KEYS,0)

    @staticmethod
    def _identity(component,key,sources):
        if not isinstance(component,str) or not component or len(component)>80:
            raise ValueError('Invalid reasoning component.')
        return {'component':component,'key_sha256':digest(key),'sources':source_binding(sources)}

    def _path(self,identity):
        # File paths never come from serialized input or component names.
        return self.directory/(digest([identity['component'],identity['key_sha256']])+'.json')

    def load(self,component,key,*,sources,check=lambda:None):
        check();identity=self._identity(component,key,sources);self.stats['reasoning_cache_reads']+=1
        path=self._path(identity)
        try:
            if path.stat().st_size>min(MAX_RECORD_BYTES,self.max_bytes):raise ValueError('Oversized local record.')
            raw=path.read_bytes()
        except (OSError,ValueError):
            self.stats['reasoning_cache_invalid']+=1;return None
        check()
        try:
            record=json.loads(raw)
            expected={'version','authority','context_sha256','identity','payload','digest'}
            if (type(record)is not dict or set(record)!=expected or type(record['version'])is not int
                    or record['version']!=VERSION or record['authority']!=AUTHORITY
                    or record['context_sha256']!=self.context_hash or record['identity']!=identity
                    or record['digest']!=digest({k:v for k,v in record.items() if k!='digest'})):
                raise ValueError('Local record identity or integrity differs.')
        except (OSError,ValueError,TypeError,UnicodeError):
            # Keep user cancellation separate from recoverable cache damage.
            self.stats['reasoning_cache_invalid']+=1;return None
        check();self.stats['reasoning_cache_hits']+=1;return record['payload']

    def save(self,component,key,payload,*,sources,check=lambda:None):
        check();identity=self._identity(component,key,sources)
        record={'version':VERSION,'authority':AUTHORITY,'context_sha256':self.context_hash,
                'identity':identity,'payload':payload}
        record['digest']=digest(record);raw=encoded(record)
        if not self.max_entries or not self.global_max_entries or len(raw)>min(MAX_RECORD_BYTES,self.max_bytes,self.global_max_bytes):return False
        path=self._path(identity);temporary=None;callback_error=None
        def poll():
            nonlocal callback_error
            try:check()
            except BaseException as exc:callback_error=exc;raise
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            try:
                if path.stat().st_size==len(raw) and path.read_bytes()==raw:
                    self.stats['reasoning_cache_unchanged']+=1;return True
            except OSError:pass
            poll()
            # Atomic replacement is the sole commit point. Scope/source checks
            # occur before writing; cancellation is polled before replacement.
            fd,temporary=tempfile.mkstemp(prefix='.reasoning-',suffix='.tmp',dir=self.directory)
            with os.fdopen(fd,'wb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            poll()
            if source_binding(sources)!=identity['sources']:raise ValueError('Reasoning source changed during save.')
            os.replace(temporary,path);temporary=None
            self.stats['reasoning_cache_writes']+=1;self.stats['reasoning_cache_write_bytes']+=len(raw)
        except OSError:
            if callback_error is not None:raise
            self.stats['reasoning_cache_write_failures']+=1;return False
        finally:
            if temporary is not None:
                try:Path(temporary).unlink()
                except OSError:pass
        self._evict(path,check);self.prune_global(current=path,check=check);return True

    def _evict(self,current,check):
        entries=[]
        for path in self.directory.glob('*.json'):
            check()
            try:info=path.stat()
            except OSError:continue
            entries.append((info.st_mtime_ns,path,info.st_size))
        total=sum(row[2] for row in entries);count=len(entries)
        for _,path,size in sorted(entries):
            if count<=self.max_entries and total<=self.max_bytes:break
            check()
            if path==current:continue
            try:path.unlink()
            except OSError:continue
            total-=size;count-=1;self.stats['reasoning_cache_evictions']+=1

    def discard(self,component,key):
        # Never called by external proof import; deletion only loses a cache.
        path=self._path({'component':component,'key_sha256':digest(key)})
        try:path.unlink()
        except FileNotFoundError:pass

    def prune_global(self,*,current=None,check=lambda:None,stale_seconds=86400):
        """Evict only managed optional records across contexts; never frontier.

        Unknown schema/source changes migrate by cold reconstruction, never by
        relabeling learned state. Old files remain optional eviction candidates.
        Symlinks/reparse paths outside this exact namespace are skipped.
        A killed writer's stale temporary file contains no committed authority.
        Concurrent eviction may lose reuse, but each individual read validates
        a complete atomic record and each writer retains its current commit.
        """
        if type(stale_seconds)not in (int,float) or stale_seconds<0:raise ValueError('Invalid stale temporary age.')
        root=self.directory.parent.resolve();entries=[];now=time.time()
        try:contexts=list(root.iterdir())
        except OSError:return
        for directory in contexts:
            check()
            if (not directory.is_dir() or directory.is_symlink() or len(directory.name)!=64
                    or any(c not in '0123456789abcdef' for c in directory.name) or directory.resolve().parent!=root):continue
            try:paths=list(directory.iterdir())
            except OSError:continue
            for path in paths:
                check()
                if path.is_symlink() or path.resolve().parent!=directory.resolve():continue
                try:info=path.stat()
                except OSError:continue
                if not path.is_file():continue
                if path.name.startswith('.reasoning-') and path.suffix=='.tmp' and now-info.st_mtime>stale_seconds:
                    try:path.unlink();self.stats['reasoning_cache_stale_temporaries']+=1
                    except OSError:pass
                elif path.suffix=='.json' and len(path.stem)==64 and all(c in '0123456789abcdef' for c in path.stem):entries.append((info.st_mtime_ns,path,info.st_size))
        total=sum(size for _,_,size in entries);count=len(entries)
        for _,path,size in sorted(entries):
            if total<=self.global_max_bytes and count<=self.global_max_entries:break
            check()
            if current is not None and path==current:continue
            try:path.unlink()
            except OSError:continue
            total-=size;count-=1;self.stats['reasoning_cache_global_evictions']+=1
