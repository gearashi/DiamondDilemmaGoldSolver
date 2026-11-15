"""Lossless transport of physical board cells and raw used-piece words."""
from pathlib import Path
from time import perf_counter
import ctypes
import hashlib
import json
import platform
import sys

import numpy as np


def encode_layout(n,lanes,depths):
    ids=np.asarray(lanes)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=n):
        raise ValueError('Invalid physical board lanes.')
    value=np.asarray(depths)
    if value.shape!=(n,) or value.dtype.kind not in 'iu':
        return None
    sizes=value[ids]
    if np.any(sizes<0) or np.any(sizes>160):
        return None
    k=len(ids)
    layout=np.empty(2+3*k,np.uint32)
    layout[:2]=(n,k)
    layout[2:2+k]=ids
    widths=(sizes.astype(np.uint32)+1)//2
    starts=layout[2+k:2+2*k]
    starts[:]=np.cumsum(widths,dtype=np.uint32)-widths
    layout[2+2*k:]=sizes
    words=int(np.sum(widths,dtype=np.uint64))
    return layout,words


def _dimensions(layout,packet):
    if (layout.ndim!=1 or layout.dtype!=np.uint32 or len(layout)<2 or packet.ndim!=1
            or packet.dtype!=np.uint32 or not layout.flags.c_contiguous or not packet.flags.c_contiguous):
        return None
    n,k=map(int,layout[:2])
    if not 0<=k<=n<=131072 or len(layout)!=2+3*k:return None
    return k


def unpack(layout,packet):
    k=_dimensions(layout,packet)
    if k is None:return None
    starts=layout[2+k:2+2*k]
    caps=layout[2+2*k:]
    widths=(caps+1)//2
    if np.any(caps>160) or not np.array_equal(starts,np.cumsum(widths,dtype=np.uint32)-widths):return None
    required=1+10*k+int(np.sum(widths,dtype=np.uint64))
    if packet.dtype!=np.uint32 or packet.shape!=(required,) or packet[0]:
        return None
    masks=packet[1:1+5*k].reshape(5,k)
    used=packet[1+5*k:1+10*k].reshape(5,k).copy()
    codes=packet[1+10*k:]
    board=np.full((160,k),-1,np.int16)
    # Bound temporary storage on the maximum-width 131,072-lane pool.
    for first in range(0,k,2048):
        last=min(k,first+2048)
        raw=np.ascontiguousarray(masks[:,first:last].T,dtype='<u4')
        active=np.unpackbits(raw.view(np.uint8),axis=1,bitorder='little')
        counts=active.sum(axis=1,dtype=np.int32)
        if np.any(counts>caps[first:last]):
            return None
        lanes,cells=np.nonzero(active)
        offsets=np.cumsum(counts,dtype=np.int32)-counts
        ranks=np.arange(len(cells),dtype=np.int32)-np.repeat(offsets,counts)
        ids=first+lanes
        halves=(codes[starts[ids]+ranks//2]>>(16*(ranks&1)))&65535
        board[cells,ids]=halves.astype(np.uint16).view(np.int16)
    return board,used


def native_unpack(library):
    """Wrap the optional compiled decoder supplied by the caller."""
    array=np.ctypeslib.ndpointer
    one=[array(dtype=np.uint32,ndim=1,flags=('C_CONTIGUOUS','ALIGNED'))]*3
    fn=library.dd_physical_board_unpack
    fn.argtypes=[ctypes.c_int,ctypes.c_int,*one,array(dtype=np.int16,ndim=2,flags=('C_CONTIGUOUS','ALIGNED')),
                 array(dtype=np.uint32,ndim=2,flags=('C_CONTIGUOUS','ALIGNED'))]
    fn.restype=ctypes.c_int
    def decode(layout,packet):
        k=_dimensions(layout,packet)
        if k is None:return None
        board=np.empty((160,k),np.int16)
        used=np.empty((5,k),np.uint32)
        if fn(k,len(packet),layout[2+k:2+2*k],layout[2+2*k:],packet,board,used):
            return board,used
        return None
    return decode


def verified_decoder(directory=None):
    root=Path(__file__).resolve().parent
    fn=Path(directory) if directory is not None else root/'.native'
    name=('r35_board_codec_decode.dll' if sys.platform=='win32' else
          'libr35_board_codec_decode.dylib' if sys.platform=='darwin' else 'libr35_board_codec_decode.so')
    try:
        manifest=json.loads((fn/'physical-board-build.json').read_bytes())
        if (not isinstance(manifest,dict) or type(manifest.get('abi')) is not int or manifest['abi']!=1 or manifest.get('library')!=name
                or manifest.get('platform')!=sys.platform or manifest.get('machine')!=platform.machine()
                or manifest.get('source_sha256')!=hashlib.sha256((root/'r35_board_codec_decode.c').read_bytes()).hexdigest()
                or manifest.get('library_sha256')!=hashlib.sha256((fn/name).read_bytes()).hexdigest()):
            return None
        return native_unpack(ctypes.CDLL(str((fn/name).resolve())))
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return None


class PhysicalBoardCodec:
    def __init__(self,gpu,decoder='auto',*,native_directory=None):
        self.gpu=gpu
        if decoder=='auto':
            self.decoder=verified_decoder(native_directory) or unpack
        elif decoder=='numpy':
            self.decoder=unpack
        elif callable(decoder):
            self.decoder=decoder
        else:
            raise ValueError("Choose 'auto', 'numpy' or an explicit physical packet decoder.")
        self.web=getattr(gpu,'backend',None)=='webgpu'
        self.pipeline=None
        self.read_bytes=self.upload_bytes=self.maps=0
        self.transfer_seconds=self.decode_seconds=0.

    def packet(self,layout,words):
        gpu=self.gpu
        k=int(layout[1])
        size=1+10*k+words
        here=Path(__file__).resolve().parent
        if not self.web:
            cp=gpu.cp
            if self.pipeline is None:
                self.pipeline=cp.RawKernel((here/'r35_board_codec.cu').read_text(encoding='utf8'),'physical_board')
            header=cp.asarray(layout)
            result=cp.empty(size,cp.uint32)
            result[:1].fill(0)
            self.pipeline(((k+63)//64,),(64,),(gpu.boards,gpu.used,header,result))
            return result.get()
        if self.pipeline is None:
            shader=gpu.gpu_device.create_shader_module(code=(here/'r35_board_codec.wgsl').read_text(encoding='utf8'))
            self.pipeline=gpu.gpu_device.create_compute_pipeline(layout='auto',compute={'module':shader,'entry_point':'physical_board'})
        usage=gpu.wgpu.BufferUsage
        native=gpu.gpu_device
        header=native.create_buffer_with_data(data=layout,usage=usage.STORAGE)
        result=None
        try:
            result=native.create_buffer(size=size*4,usage=usage.STORAGE|usage.COPY_SRC)
            group=native.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),entries=[
                {'binding':i,'resource':{'buffer':buf}} for i,buf in enumerate((gpu.buffers[0],header,result))])
            enc=native.create_command_encoder()
            comp=enc.begin_compute_pass()
            comp.set_pipeline(self.pipeline)
            comp.set_bind_group(0,group)
            comp.dispatch_workgroups((k+63)//64)
            comp.end()
            gpu.queue.submit([enc.finish()])
            return np.frombuffer(gpu.queue.read_buffer(result),dtype=np.uint32).copy()
        finally:
            header.destroy()
            if result is not None:result.destroy()

    def capture(self,fields,lanes,*,mode=None,installed=None):
        """Return original board and used columns, or None when a reservation is too small."""
        self.maps=self.upload_bytes=self.read_bytes=0
        self.transfer_seconds=self.decode_seconds=0.
        encoded=encode_layout(self.gpu.n,lanes,fields['depths'])
        if encoded is None:return None
        layout,words=encoded
        if len(lanes)==0:
            return np.empty((160,0),np.int16),np.empty((5,0),np.uint32)
        before=perf_counter()
        payload=self.packet(layout,words)
        self.transfer_seconds=perf_counter()-before
        self.read_bytes=payload.nbytes
        self.upload_bytes=layout.nbytes
        self.maps=1
        if self.web:self.gpu.readback_bytes+=self.read_bytes
        before=perf_counter()
        out=self.decoder(layout,payload)
        self.decode_seconds=perf_counter()-before
        return out
