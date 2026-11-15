"""Losslessly transfer native board changes relative to the last frozen CPU snapshot."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np

POPCOUNT=np.array([v.bit_count() for v in range(256)],np.uint8)


def _values(raw,base,j,width):
    bit=j*width
    at=base+(bit>>5)
    shift=bit&31
    # The decoder owns a final zero padding word; bit slices may cross a word.
    pair=raw[at].astype(np.uint64)|(raw[at+1].astype(np.uint64)<<32)
    return ((pair>>shift)&((1<<width)-1)).astype(np.uint32)


def _decode(prior,lanes,summary,layout,payload,decoder=None):
    mode,counts,bits,used,starts,ends,_=layout
    rows=prior[0].copy() if np.any(summary&255) else prior[0]
    masks=prior[1].copy() if np.any(used) else prior[1]
    if decoder is not None:
        ok=decoder(rows.shape[1],len(lanes),len(payload),np.ascontiguousarray(lanes,np.int32),
                   layout[-1],starts,ends,payload,rows,masks)
        if not ok:
            raise ValueError('Invalid physical board delta payload.')
        return rows,masks
    raw=np.empty(len(payload)+1,np.uint32)
    raw[:-1]=payload
    raw[-1]=0
    reset=lanes[(mode==1)|(mode==3)]
    if len(reset):
        rows[:,reset]=-1
    for b in (9,16):
        for m in range(5):
            sel=np.flatnonzero((bits==b)&(mode==m)&(counts>0))
            if not len(sel):
                continue
            ids=lanes[sel]
            sizes=counts[sel]
            base=starts[sel]
            if m in (2,3):
                used_rows=raw[base[:,None]+np.arange(5)]
                mask_bytes=np.ascontiguousarray(used_rows).view(np.uint8)
                observed=POPCOUNT[mask_bytes].sum(axis=1,dtype=np.uint32)
                if not np.array_equal(observed,sizes):
                    raise ValueError('Physical change mask disagrees with its value count.')
                idx=np.zeros(len(sel),np.uint32)
                for row in range(160):
                    active=np.flatnonzero((used_rows[:,row//32]>>(row%32))&1)
                    if len(active):
                        codes=_values(raw,base[active]+5,idx[active],b)
                        if b==9:
                            if np.any(codes>480):
                                raise ValueError('Reserved compact board code.')
                            codes=np.where(codes==480,65535,codes)
                        rows[row,ids[active]]=codes.astype(np.uint16).view(np.int16)
                        idx[active]+=1
            else:
                width=b+8 if m<2 else b
                last_cell=np.full(len(sel),-1,np.int32) if m<2 else None
                for j in range(int(sizes.max())):
                    active=np.flatnonzero(sizes>j)
                    token=_values(raw,base[active],j,width)
                    cell=token>>b if m<2 else j
                    if m<2:
                        if np.any(cell>=160) or np.any(cell<=last_cell[active]):
                            raise ValueError('Invalid physical row order.')
                        last_cell[active]=cell
                    codes=token&((1<<b)-1)
                    if b==9:
                        if np.any(codes>480):
                            raise ValueError('Reserved compact board code.')
                        codes=np.where(codes==480,65535,codes)
                    rows[cell,ids[active]]=codes.astype(np.uint16).view(np.int16)
    pos=ends-POPCOUNT[used]
    for row in range(5):
        sel=np.flatnonzero(used&(1<<row))
        if len(sel):
            masks[row,lanes[sel]]=raw[pos[sel]]
            pos[sel]+=1
    return rows,masks


def _directory(k,header,packet):
    BoardDelta._packet_size(k,header)
    failure,dirty,payload_words,directory_words,form=map(int,header)
    if failure or not 0<=dirty<=k or form not in (0,1,2):
        raise ValueError('Invalid native board packet header.')
    if directory_words!=(dirty*form if form else (k+1)//2) or len(packet)!=directory_words+payload_words:
        raise ValueError('Invalid native board packet dimensions.')
    directory=packet[:directory_words]
    if not form:
        # Split explicitly, independent of CPU byte order.
        tickets=np.empty(2*len(directory),np.uint32)
        tickets[::2]=directory&65535
        tickets[1::2]=directory>>16
        indices=np.flatnonzero(tickets[:k])
        tickets=tickets[indices]
    elif form==1:
        indices=directory>>16
        tickets=directory&65535
    else:
        indices=directory[::2]
        tickets=directory[1::2]
    if (len(indices)!=dirty or np.any(tickets==0) or np.any(tickets>51520)
            or (len(indices) and (indices[-1]>=k or np.any(indices[1:]<=indices[:-1])))):
        raise ValueError('Invalid native board change directory.')
    val=tickets-1
    used=val&31
    choice=(val>>5)%10
    mode=choice%5
    bits=np.where(choice>=5,16,9).astype(np.uint32)
    counts=val//320
    size=((np.where(mode<2,bits+8,bits)*counts+31)//32)+np.where((mode==2)|(mode==3),5,0).astype(np.uint32)+POPCOUNT[used]
    ends=np.cumsum(size,dtype=np.uint32)
    if (int(ends[-1]) if len(ends) else 0)!=payload_words or np.any(counts>160) or np.any((mode==4)&(counts!=160)):
        raise ValueError('Invalid packed native board widths.')
    flags=mode|((bits==16).astype(np.uint32)<<3)|(used<<4)|(counts<<10)
    layout=mode,counts,bits,used,ends-size,ends,flags
    board_changed=((mode!=0)|(counts!=0)).astype(np.uint32)
    return indices,board_changed,layout,packet[directory_words:]


def verified_decoder(directory=None):
    """Load the optional local decoder only for the current source and architecture."""
    import ctypes
    import hashlib
    import json
    import platform
    import sys
    from build_native import library_name
    root=Path(__file__).resolve().parent
    directory=Path(directory) if directory else root/'.native'
    name=library_name('dfs_checkpoint_decode')
    try:
        manifest=json.loads((directory/'board-build.json').read_bytes())
        if (not isinstance(manifest,dict) or type(manifest.get('abi')) is not int or manifest['abi']!=1
                or manifest.get('library')!=name or manifest.get('platform')!=sys.platform
                or manifest.get('machine')!=platform.machine()
                or manifest.get('source_sha256')!=hashlib.sha256((root/'dfs_checkpoint_decode.c').read_bytes()).hexdigest()
                or manifest.get('library_sha256')!=hashlib.sha256((directory/name).read_bytes()).hexdigest()):
            return None
        fn=ctypes.CDLL(str((directory/name).resolve())).dd_apply_board_delta
        array=np.ctypeslib.ndpointer
        fn.argtypes=[ctypes.c_int]*3+[array(dtype=np.int32,ndim=1,flags=('C_CONTIGUOUS','ALIGNED'))]+[
            array(dtype=np.uint32,ndim=1,flags=('C_CONTIGUOUS','ALIGNED'))]*4+[
            array(dtype=d,ndim=2,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')) for d in (np.int16,np.uint32)]
        fn.restype=ctypes.c_int
        return fn
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return None


class BoardDelta:
    def __init__(self,gpu,*,native=False,device_plan=True):
        self.gpu=gpu
        self.web=getattr(gpu,'backend',None)=='webgpu'
        self.mirror=self.kernels=self.last=None
        self.decoder=None
        if native:
            self.decoder=verified_decoder()
        self.read_bytes=self.upload_bytes=self.maps=0
        self.distribution={}
        self.plan_cache=None
        if self.web:
            gpu._resources.callback(self.close)

    def close(self):
        if self.mirror is not None and self.web:
            self.mirror.destroy()
        self.mirror=self.last=None
        if self.plan_cache is not None and self.web:
            self.plan_cache[1].destroy()
        self.plan_cache=None

    def seed(self,fields):
        gpu=self.gpu
        if self.web:
            if self.mirror is None:
                self.mirror=gpu.gpu_device.create_buffer(size=gpu.n*660,
                    usage=gpu.wgpu.BufferUsage.STORAGE|gpu.wgpu.BufferUsage.COPY_DST)
            enc=gpu.gpu_device.create_command_encoder()
            enc.copy_buffer_to_buffer(gpu.buffers[0],0,self.mirror,0,gpu.n*660)
            gpu.queue.submit([enc.finish()])
        elif self.mirror is None:
            self.mirror=(gpu.boards.copy(),gpu.used.copy())
        else:
            gpu.cp.copyto(self.mirror[0],gpu.boards)
            gpu.cp.copyto(self.mirror[1],gpu.used)
        self.last=fields['boards'],fields['used']

    def _plan(self,lanes,header):
        if self.plan_cache is not None:
            cached,desc=self.plan_cache
            if np.array_equal(cached,lanes):
                return desc
            if self.web:
                desc.destroy()
            self.plan_cache=None
        gpu=self.gpu
        if self.web:
            usage=gpu.wgpu.BufferUsage
            desc=gpu.gpu_device.create_buffer(size=header.nbytes,usage=usage.STORAGE|usage.COPY_DST)
            try:
                gpu.queue.write_buffer(desc,0,header)
            except BaseException:
                desc.destroy()
                raise
        else:
            desc=gpu.cp.asarray(header)
        self.upload_bytes+=header.nbytes
        self.plan_cache=(lanes.copy(),desc)
        return desc

    def _device_packet(self,lanes,header):
        gpu=self.gpu
        k=len(lanes)
        groups=(k+127)//128
        desc=self._plan(lanes,header)
        names=('price_groups','scan_groups','pack_groups')
        if self.web:
            if self.kernels is None:
                src=(Path(__file__).resolve().parent/'dfs_checkpoint_delta.wgsl').read_text(encoding='utf8')
                shader=gpu.gpu_device.create_shader_module(code=src)
                self.kernels={key:gpu.gpu_device.create_compute_pipeline(layout='auto',compute={'module':shader,'entry_point':key})
                              for key in names}
            usage=gpu.wgpu.BufferUsage
            device=gpu.gpu_device
            def dispatch(encoder,name,nb,bindings):
                pipe=self.kernels[name]
                group=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[
                    {'binding':i,'resource':{'buffer':b}} for i,b in bindings])
                comp=encoder.begin_compute_pass()
                comp.set_pipeline(pipe);comp.set_bind_group(0,group)
                comp.dispatch_workgroups(nb);comp.end()
            with ExitStack() as resources:
                work=device.create_buffer(size=4*(9*k+2*groups),usage=usage.STORAGE)
                resources.callback(work.destroy)
                head=device.create_buffer(size=20,usage=usage.STORAGE|usage.COPY_SRC)
                resources.callback(head.destroy)
                enc=device.create_command_encoder()
                dispatch(enc,names[0],groups,[(0,gpu.buffers[0]),(1,self.mirror),(2,desc),(4,work)])
                dispatch(enc,names[1],1,[(2,desc),(3,head),(4,work)])
                gpu.queue.submit([enc.finish()])
                meta=np.frombuffer(gpu.queue.read_buffer(head,0,20),np.uint32).copy()
                self.maps+=1;self.read_bytes+=20;gpu.readback_bytes+=20
                size=self._packet_size(k,meta)
                if not meta[1]:
                    return None
                out=device.create_buffer(size=4*(5+size),usage=usage.STORAGE|usage.COPY_SRC|usage.COPY_DST)
                resources.callback(out.destroy)
                enc=device.create_command_encoder()
                enc.copy_buffer_to_buffer(head,0,out,0,20)
                bindings=[(0,gpu.buffers[0]),(2,desc),(3,out),(4,work)]
                dispatch(enc,names[2],(k+63)//64,bindings)
                gpu.queue.submit([enc.finish()])
                raw=np.frombuffer(gpu.queue.read_buffer(out,0,4*(5+size)),np.uint32).copy()
                gpu.readback_bytes+=raw.nbytes
        else:
            cp=gpu.cp
            if self.kernels is None:
                src=(Path(__file__).resolve().parent/'dfs_checkpoint_delta.cu').read_text(encoding='utf8')
                mod=cp.RawModule(code=src)
                self.kernels={name:mod.get_function(name) for name in names}
            work=cp.empty(9*k+2*groups,cp.uint32)
            head=cp.empty(5,cp.uint32)
            self.kernels[names[0]]((groups,),(128,),(gpu.boards,gpu.used,*self.mirror,desc,work))
            self.kernels[names[1]]((1,),(1,),(desc,work,head))
            meta=head.get()
            self.maps+=1;self.read_bytes+=20
            size=self._packet_size(k,meta)
            if not meta[1]:
                return None
            out=cp.empty(5+size,cp.uint32)
            cp.copyto(out[:5],head)
            self.kernels[names[2]](((k+63)//64,),(64,),(gpu.boards,gpu.used,*self.mirror,desc,work,out))
            raw=out.get()
        self.maps+=1;self.read_bytes+=raw.nbytes
        if not np.array_equal(meta,raw[:5]):
            raise ValueError('Native board changed while packing its checkpoint.')
        return _directory(k,raw[:5],raw[5:])

    @staticmethod
    def _packet_size(k,meta):
        failure,dirty,words,directory,form=map(int,meta)
        sparse=1 if k<=65536 else 2
        expected_form=0 if (k+1)//2<=dirty*sparse else sparse
        if (failure or dirty>k or words>85*dirty or form!=expected_form
                or directory!=(dirty*form if form else (k+1)//2)):
            raise ValueError('Native board packet exceeds its physical bounds.')
        return words+directory

    def capture(self,lanes):
        self.read_bytes=self.upload_bytes=self.maps=0
        self.distribution={}
        if self.last is None:
            return None
        lanes=np.asarray(lanes)
        if lanes.ndim!=1 or lanes.dtype.kind not in 'iu' or (len(lanes) and (
                lanes[0]<0 or lanes[-1]>=self.gpu.n or np.any(lanes[1:]<=lanes[:-1]))):
            raise ValueError('Checkpoint lanes must be sorted, distinct and within the replica pool.')
        k=len(lanes)
        if not k:
            return self.last
        if int(lanes[-1])-int(lanes[0])+1==k:
            header=np.array([self.gpu.n,k,0x80000000|int(lanes[0])],np.uint32)
        else:
            header=np.r_[np.array([self.gpu.n,k,0],np.uint32),lanes.astype(np.uint32)]
        previous=self.last
        self.last=None
        packet=self._device_packet(lanes,header)
        if packet is None:
            self.last=previous
        else:
            ids,changed,layout,payload=packet
            self.distribution={str(i):int((layout[0]==i).sum()) for i in range(5)}
            self.last=_decode(previous,lanes[ids],changed,layout,payload,self.decoder)
        return self.last
