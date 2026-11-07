"""Bounded immutable GPU checkpoint generations and conservative recovery.

No garbage collection runs: delayed asynchronous consumers retain valid refs.
At the logical-byte/file cap, callers save a self-contained legacy checkpoint.
A reference resolves only its named generation (or explicit predecessor when
recovery is requested), never an unrelated or newer HEAD generation.
"""
from __future__ import annotations
import base64,copy,json,os,uuid
from pathlib import Path
from constraint_work import digest
from gpu_checkpoint_codec_round25 import canonical_bytes,sha,validate_spec,checked_blob

DEFAULT_STORE_BYTES=512*1024*1024
MAX_MANIFEST_BYTES=64*1024*1024
MAX_FILES=32768
MAX_HEAD_BYTES=16384
REF_KEY='gpu_checkpoint_reference'

class StoreCapacity(RuntimeError):pass
class StoreFailure(RuntimeError):pass
class ReferenceMismatch(ValueError):pass

def _read(path,limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:raise ValueError('Missing, linked, or oversized GPU checkpoint object.')
    with path.open('rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('GPU checkpoint object exceeds read bound.')
    return data

def _path(directory,name):
    path=directory/name
    if path.resolve().parent!=directory.resolve():raise ValueError('GPU checkpoint path escapes its store.')
    return path

def _hash(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

def _identity(directory):
    directory=Path(directory).resolve();value=json.loads(_read(_path(directory,'identity.json'),4096))
    if (not isinstance(value,dict) or set(value)!={'version','store_id','scope_digest'} or type(value['version'])is not int or value['version']!=1
            or not isinstance(value['store_id'],str) or len(value['store_id'])!=32 or any(x not in '0123456789abcdef' for x in value['store_id'])
            or not _hash(value['scope_digest'])):raise ValueError('Invalid GPU checkpoint store identity.')
    return directory,value

def is_reference(payload):return isinstance(payload,dict) and REF_KEY in payload

def _check_reference(ref,identity):
    if (not isinstance(ref,dict) or set(ref)!={REF_KEY,'store_id','scope_digest','generation','previous','frontier_digest','digest'}
            or type(ref[REF_KEY])is not int or ref[REF_KEY]!=1 or ref['digest']!=digest({k:v for k,v in ref.items() if k!='digest'})
            or ref['store_id']!=identity['store_id'] or ref['scope_digest']!=identity['scope_digest']
            or not _hash(ref['generation']) or ref['previous'] is not None and not _hash(ref['previous']) or not _hash(ref['frontier_digest'])):
        raise ValueError('GPU checkpoint reference or lineage differs.')

def _manifest(directory,identity,generation):
    if not _hash(generation):raise ValueError('Invalid generation identity.')
    manifests=(directory/'manifests').resolve()
    if manifests.parent!=directory:raise ValueError('GPU manifest directory escapes store.')
    raw=_read(_path(manifests,generation+'.json'),MAX_MANIFEST_BYTES)
    if sha(raw)!=generation:raise ValueError('GPU generation checksum differs.')
    value=json.loads(raw)
    if (not isinstance(value,dict) or set(value)!={'version','store_id','scope_digest','previous','payload'}
            or type(value['version'])is not int or value['version']!=1 or value['store_id']!=identity['store_id']
            or value['scope_digest']!=identity['scope_digest'] or value['previous'] is not None and not _hash(value['previous'])):
        raise ValueError('GPU generation lineage differs.')
    payload=value['payload']
    if (not isinstance(payload,dict) or payload.get('digest')!=digest({k:v for k,v in payload.items() if k!='digest'})
            or digest(payload.get('scope'))!=identity['scope_digest']):raise ValueError('GPU generation runner state differs.')
    frontier=payload.get('frontier')
    if (not isinstance(frontier,dict) or frontier.get('digest')!=digest({k:v for k,v in frontier.items() if k!='digest'})
            or frontier.get('scope')!=payload['scope']):raise ValueError('GPU generation frontier binding differs.')
    return value

def _reference(identity,generation,manifest):
    ref={REF_KEY:1,'store_id':identity['store_id'],'scope_digest':identity['scope_digest'],'generation':generation,
         'previous':manifest['previous'],'frontier_digest':manifest['payload']['frontier']['digest']}
    ref['digest']=digest(ref);return ref

def checkpoint_frontier(payload,directory):
    """Read hash-bound host metadata only; never decompress native arrays."""
    if not is_reference(payload):return copy.deepcopy(payload.get('frontier')) if isinstance(payload,dict) else None
    directory,identity=_identity(directory);_check_reference(payload,identity)
    manifest=_manifest(directory,identity,payload['generation'])
    if _reference(identity,payload['generation'],manifest)!=payload:raise ValueError('GPU generation reference metadata differs.')
    return copy.deepcopy(manifest['payload']['frontier'])

def _native_scope(native,payload):
    validate_spec(native);binding=native.get('binding');scope=payload.get('scope',{})
    if not isinstance(binding,dict) or any(binding.get(key)!=scope.get(key) for key in ('fingerprint','target','original')):raise ValueError('Native state and whole generation scope differ.')

def _materialize(directory,payload):
    result=copy.deepcopy(payload);native=result.get('native')
    if native is not None:
        _native_scope(native,payload);chunks=(directory/'chunks').resolve()
        if chunks.parent!=directory:raise ValueError('GPU chunk directory escapes store.')
        for column in native['columns']:
            if 'data' in column:raise ValueError('Immutable manifest unexpectedly contains inline chunk data.')
            raw=checked_blob(column,_read(_path(chunks,column['sha256']+'.bin'),column['compressed_bytes']))
            column['data']=base64.b64encode(raw).decode('ascii')
    result['digest']=digest({k:v for k,v in result.items() if k!='digest'});return result

def resolve_checkpoint(payload,directory,*,allow_recovery=False):
    if type(allow_recovery)is not bool:raise ValueError('Recovery policy must be boolean.')
    if not is_reference(payload):return payload
    directory,identity=_identity(directory);_check_reference(payload,identity)
    generation=payload['generation'];previous=payload['previous'];skipped=0
    while True:
        try:
            manifest=_manifest(directory,identity,generation)
            if skipped==0 and _reference(identity,generation,manifest)!=payload:raise ReferenceMismatch('GPU generation reference metadata differs.')
            result=_materialize(directory,manifest['payload'])
            result['checkpoint_origin']=_reference(identity,generation,manifest)
            result['digest']=digest({k:v for k,v in result.items() if k!='digest'})
            if skipped:
                result['checkpoint_recovery']={'requested':payload['generation'],'restored':generation,'generations_rolled_back':skipped,
                    'note':'Recovered the complete known previous generation; work after it may repeat.'}
                result['digest']=digest({k:v for k,v in result.items() if k!='digest'})
            return result
        except ReferenceMismatch:raise
        except (ValueError,OSError,json.JSONDecodeError):
            if not allow_recovery or previous is None or skipped>=8:raise
            generation=previous;skipped+=1
            # Only the explicit predecessor chain is eligible. A missing older
            # manifest provides no trustworthy link to continue farther back.
            older=_manifest(directory,identity,generation);previous=older['previous']

def _atomic(path,raw):
    if path.parent.resolve()!=path.parent:raise ValueError('Checkpoint write directory is linked.')
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        try:
            descriptor=os.open(str(path.parent),os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        except OSError:pass # Directory fsync is not available on every Windows filesystem.
    finally:
        if temporary.exists():temporary.unlink()

class CheckpointStore:
    """One writer per run directory, serialized by the app process lock.

    This helper does not provide an interprocess lock for direct callers.
    """
    def __init__(self,directory,scope,*,max_bytes=DEFAULT_STORE_BYTES,previous=None):
        if type(max_bytes)is not int or max_bytes<8192:raise ValueError('Checkpoint store cap must be at least8192 bytes.')
        self.directory=Path(directory).resolve();self.scope_digest=digest(scope);self.max_bytes=max_bytes;self.cache={}
        self.stats={'generations':0,'written_chunks':0,'written_bytes':0,'reused_objects':0};self.status='ready';self.reason=''
        self.directory.mkdir(parents=True,exist_ok=True)
        identity_path=_path(self.directory,'identity.json')
        if not identity_path.exists():_atomic(identity_path,canonical_bytes({'version':1,'store_id':uuid.uuid4().hex,'scope_digest':self.scope_digest}))
        _,self.identity=_identity(self.directory)
        if self.identity['scope_digest']!=self.scope_digest:raise ValueError('Checkpoint store belongs to another exact scope.')
        for name in ('chunks','manifests'):
            child=self.directory/name
            if child.resolve()!=child:raise ValueError('Checkpoint subdirectory is linked or escapes store.')
            child.mkdir(exist_ok=True)
        self.previous=None
        if previous is not None:
            _check_reference(previous,self.identity);manifest=_manifest(self.directory,self.identity,previous['generation'])
            if _reference(self.identity,previous['generation'],manifest)!=previous:raise ValueError('Resume origin differs from known generation.')
            self.previous=copy.deepcopy(previous)
    def _usage(self):
        total=0;count=0
        for folder in (self.directory,self.directory/'chunks',self.directory/'manifests'):
            if folder.resolve()!=folder:raise StoreFailure('Linked GPU checkpoint directory.')
            for path in folder.iterdir():
                if path.is_dir():
                    if folder!=self.directory or path.name not in ('chunks','manifests'):raise StoreFailure('Unexpected GPU checkpoint subdirectory.')
                    continue
                if path.is_symlink() or path.resolve().parent!=folder.resolve():raise StoreFailure('Linked GPU checkpoint object.')
                total+=path.stat().st_size;count+=1
                if count>MAX_FILES or total>self.max_bytes:raise StoreCapacity('GPU checkpoint retention cap reached.')
        return total,count
    def _head(self):
        path=_path(self.directory,'HEAD.json')
        if not path.exists():return None
        value=json.loads(_read(path,MAX_HEAD_BYTES))
        if not isinstance(value,dict) or set(value)!={'version','current','previous'} or type(value['version'])is not int or value['version']!=1:raise StoreFailure('Invalid GPU checkpoint journal.')
        _check_reference(value['current'],self.identity)
        if value['previous'] is not None:_check_reference(value['previous'],self.identity)
        return value['current']
    def write(self,payload,blobs):
        """Publish immutable chunks, then complete manifest, then atomic HEAD.

        All old objects are retained. No reference is returned before HEAD is
        durable. Failure can leave unreachable objects; they count toward cap.
        """
        previous=self.previous;body=copy.deepcopy(payload)
        body.pop('checkpoint_origin',None);body.pop('checkpoint_recovery',None)
        body['digest']=digest({k:v for k,v in body.items() if k!='digest'})
        if digest(body.get('scope'))!=self.scope_digest:raise ValueError('Runner scope differs from checkpoint store.')
        native=body.get('native');needed={}
        if native is not None:
            _native_scope(native,body)
            for column in native['columns']:
                if 'data' in column:raise ValueError('Store requires referenced native chunks.')
                key=column['sha256'];raw=checked_blob(column,blobs[key]);needed[key]=raw
        manifest={'version':1,**{k:self.identity[k] for k in ('store_id','scope_digest')},'previous':previous['generation'] if previous else None,'payload':body}
        encoded=canonical_bytes(manifest)
        if len(encoded)>MAX_MANIFEST_BYTES:raise StoreCapacity('GPU host generation exceeds manifest cap.')
        generation=sha(encoded);ref=_reference(self.identity,generation,manifest)
        head=canonical_bytes({'version':1,'current':ref,'previous':previous})
        if len(head)>MAX_HEAD_BYTES:raise StoreCapacity('GPU journal exceeds metadata cap.')
        used,count=self._usage();new={}
        for key,raw in needed.items():
            path=_path(self.directory/'chunks',key+'.bin')
            if path.exists():
                try:existing=_read(path,len(raw))
                except (ValueError,OSError) as exc:raise StoreFailure('Existing immutable GPU chunk is unreadable.') from exc
                if existing!=raw:raise StoreFailure('Existing immutable GPU chunk is corrupt.')
                self.stats['reused_objects']+=1
            else:new[key]=raw
        required=sum(map(len,new.values()))+len(encoded)+len(head)+4096
        if used+required>self.max_bytes or count+len(new)+2>MAX_FILES:raise StoreCapacity('GPU checkpoint retention cap reached; preserve existing references.')
        try:
            for key,raw in new.items():_atomic(_path(self.directory/'chunks',key+'.bin'),raw);self.stats['written_chunks']+=1;self.stats['written_bytes']+=len(raw)
            manifest_path=_path(self.directory/'manifests',generation+'.json')
            if manifest_path.exists():
                if _read(manifest_path,MAX_MANIFEST_BYTES)!=encoded:raise StoreFailure('Immutable generation is corrupt.')
            else:_atomic(manifest_path,encoded)
            _atomic(_path(self.directory,'HEAD.json'),head)
        except (OSError,ValueError) as exc:raise StoreFailure('GPU checkpoint write failed; previous references remain retained.') from exc
        self.stats['written_bytes']+=len(encoded)+len(head);self.stats['generations']+=1
        self.previous=copy.deepcopy(ref);self.status='incremental';return ref
