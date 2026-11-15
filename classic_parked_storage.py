"""Keep a portable base for parked classic stacks while saving newer active generations."""
from collections import ChainMap
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re

import numpy as np


REFERENCE = 'exact_parked_parent'
OFFSET = 'exact_parked_offset'
COUNT = 'exact_parked_count'
PAUSED = 'exact_paused_'
BANK_FILE = re.compile(r'parked-bank-([0-9a-f]{64})[.]npz')
MIN_PAUSED = 16
MIN_BANK_BYTES = 4096


def bank_path(checkpoint, sha):
    if not isinstance(sha,str) or re.fullmatch(r'[0-9a-f]{64}',sha) is None:
        raise ValueError('Invalid parked bank digest.')
    return checkpoint.with_name('parked-bank-'+sha+'.npz')


def _number(value):
    value=np.asarray(value)
    if value.ndim or value.dtype.kind not in 'iu' or not 0 <= int(value) <= 131072:
        raise ValueError('Invalid parked bank bounds.')
    return int(value)


@contextmanager
def restore_parked(checkpoint, head):
    """Expose the same full NumPy field mapping as an ordinary self-contained archive."""
    markers={REFERENCE,OFFSET,COUNT}.intersection(head.files)
    if not markers:
        yield head
        return
    if (len(markers)!=3 or any(k.startswith(PAUSED) for k in head.files)
            or str(head['exact_terminal']) or 'exact_source_origin' not in head.files):
        raise ValueError('Incomplete or conflicting parked checkpoint header.')
    sha=str(head[REFERENCE])
    path=bank_path(Path(checkpoint),sha)
    if path.resolve().parent != path.parent.resolve():
        raise ValueError('Parked checkpoint base is outside its profile.')
    offset,count=_number(head[OFFSET]),_number(head[COUNT])
    if not count:
        raise ValueError('Detached parked bank is empty.')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:
            raise ValueError('Parked checkpoint base differs from its digest.')
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as base:
            if REFERENCE in base.files or any(str(base[k])!=str(head[k]) for k in
                    ('exact_source_origin','exact_source_binding','exact_frontier_sha256','exact_data_sha256')):
                raise ValueError('Parked checkpoint base belongs to another search.')
            ids=base['exact_paused_job_ids']
            if ids.ndim!=1 or ids.dtype!=np.int64 or len(ids)!=offset+count:
                raise ValueError('Parked checkpoint base does not contain the requested suffix.')
            # Read each compressed bank member once. Copy a consumed suffix so it
            # does not retain already discharged lanes during the resumed search.
            fields={'exact_paused_job_ids':ids[offset:].copy() if offset else ids}
            for key in base.files:
                if not key.startswith(PAUSED) or key in fields:
                    continue
                value=base[key]
                axis=0 if value.ndim==1 or key=='exact_paused_prefix_codes' else 1
                if value.ndim not in (1,2) or value.shape[axis]!=len(ids):
                    raise ValueError('Parked checkpoint field has a different lane count.')
                suffix=value[offset:] if axis==0 else value[:,offset:]
                fields[key]=suffix.copy() if offset else value
            yield ChainMap(fields,head)


class ParkedCheckpointWriter:
    """Used only by the profile's single writer under the solver lock."""
    def __init__(self, path, write):
        self.path=Path(path)
        self.write_archive=write
        self.base=None
        self.observed_bank=None
        self.intervals=0
        self.linked=None
        self.link_stamp=None
        self.supports_links=True
        self.swept=False

    def _retire(self, keep):
        if self.swept and self.linked==keep:
            return
        clean=True
        for path in self.path.parent.glob('parked-bank-*.npz'):
            if BANK_FILE.fullmatch(path.name) and path != keep:
                try:
                    path.unlink()
                except OSError:
                    clean=False  # An open Windows reader can delay retirement.
        self.linked=keep
        self.swept=clean

    def _link(self, sha):
        path=bank_path(self.path,sha)
        if path != self.linked:
            self.swept=False
            try:
                os.link(self.path,path)
            except FileExistsError:
                pass
            except OSError:
                # Filesystems without hard links keep complete, portable checkpoints.
                self.supports_links=False
                return None
        try:
            if path.resolve().parent != self.path.parent.resolve():
                return None
            st=path.stat()
            stamp=(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            if path==self.linked and stamp==self.link_stamp:
                return path
            with path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:
                    return None
        except OSError:
            # In-memory ownership is still intact. Publish the complete current capture.
            return None
        self.link_stamp=stamp
        return path

    def write(self, packet):
        fields,bank,offset,interval=packet
        count=len(fields['exact_paused_job_ids'])
        if self.observed_bank is not bank:
            self.observed_bank=bank
            self.intervals=0
        self.intervals+=int(interval)
        eligible=interval and count>=MIN_PAUSED and self.supports_links
        if eligible and self.base is not None:
            source,start,sha=self.base
            if source is bank and offset>=start and len(bank.ids)-offset==count:
                path=self._link(sha)
                if path is not None:
                    active={k:v for k,v in fields.items() if not k.startswith(PAUSED)}
                    active.update({REFERENCE:np.array(sha),OFFSET:np.int64(offset-start),COUNT:np.int64(count)})
                    result=self.write_archive(self.path,active)
                    self._retire(path)
                    return {**result,'parked_bank_external':True,'parked_lanes_reused':count}
        result=self.write_archive(self.path,fields)
        # A same-generation stop can cost one more full write. Admit small
        # compressed shares only after observing enough actual interval saves.
        price=result.get('parked_member_bytes',0)
        horizon=min(16,max(3,self.intervals))
        pays_back=price>=MIN_BANK_BYTES and price*horizon>=result['file_bytes']
        self.base=(bank,offset,result['sha256']) if eligible and pays_back and self.supports_links else None
        self._retire(None)
        return {**result,'parked_bank_external':False,'parked_lanes_reused':0}

    def portable(self):
        """Finish a previously published interval generation without rereading any GPU lanes."""
        import systematic_evidence as proof
        with np.load(self.path,allow_pickle=False) as head:
            proof.check_receipt(self.path.parent.parent,str(head['exact_source_origin']),
                int(head['exact_generation']),self.path,str(head['exact_source_binding']),parked=True)
            with restore_parked(self.path,head) as archive:
                fields={k:archive[k] for k in archive if k not in (REFERENCE,OFFSET,COUNT)}
        result=self.write_archive(self.path,fields)
        self.base=None
        self._retire(None)
        return {**result,'parked_bank_external':False,'parked_lanes_reused':0}
