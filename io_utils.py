"""Atomic snapshots with Windows-compatible concurrent readers."""
from pathlib import Path
import json,os,threading,time

def replace_with_retry(source,target,timeout=2.0):
    deadline=time.monotonic()+timeout;delay=.002
    while True:
        try:os.replace(source,target);return
        except PermissionError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None)not in(5,32,33)or time.monotonic()>=deadline:raise
            time.sleep(delay);delay=min(delay*2,.05)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
        replace_with_retry(tmp,path)
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:pass

def read_json_shared(path):
    """Allow atomic replacement while this reader has the old snapshot open."""
    if os.name!='nt':
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    import ctypes,msvcrt
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    handle=create(str(Path(path)),0x80000000,1|2|4,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:close(handle);raise
    with os.fdopen(fd,'r',encoding='utf-8-sig')as stream:return json.load(stream)
