"""Adapter for the optional bounded C frame-span decoder."""
import ctypes
import numpy as np


def native_merge(library):
    fn=library.dd_merge_frame_spans
    array=np.ctypeslib.ndpointer
    one=array(dtype=np.uint32,ndim=1,flags=('C_CONTIGUOUS','ALIGNED'))
    two=array(dtype=np.uint16,ndim=2,flags=('C_CONTIGUOUS','ALIGNED'))
    fn.argtypes=[ctypes.c_uint]*3+[one]*3+[two]*4
    fn.restype=ctypes.c_int
    def merge(fields,ids,marks,widths,packet):
        del widths
        n=fields[0].shape[1]
        if (any(f.shape!=(160,n) for f in fields) or marks.shape!=ids.shape or
                packet.size>np.iinfo(np.uint32).max or len(ids)>n):
            raise ValueError('Invalid physical frame arrays.')
        if not fn(n,len(ids),len(packet),ids,marks,packet,*fields):
            raise ValueError('Invalid physical frame bounds or packet length.')
    return merge
