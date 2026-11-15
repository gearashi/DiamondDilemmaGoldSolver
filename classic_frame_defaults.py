"""Recover full physical frame arrays from their checked nondefault values on the device."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np

FIELDS=('firsts','lengths','cursors','shifts')
DEFAULTS=(0,0,65535,0)
META_WORDS=160*3


def _prices(n,meta):
    """Row directory modes and payload size, in u32 words."""
    meta=np.asarray(meta)
    if meta.shape!=(160,3):raise ValueError('Invalid physical frame summary size.')
    count,low,high=meta.astype(np.int64).T
    empty=count==0
    if (np.any(count<0) or np.any(empty&((low!=n)|(high!=0))) or
            np.any((~empty)&((low<0)|(high>n)|(low>=high)|(count>high-low)))):
        raise ValueError('Inconsistent physical frame summary.')
    span=np.where(empty,0,high-low)
    mask=np.where(empty,0,(high-1)//32-low//32+1)
    prices=np.stack((2*span,mask+2*count,3*count))
    mode=np.where(empty,0,1+np.argmin(prices,axis=0))
    directory=np.where(mode==2,mask,np.where(mode==3,count,0))
    sizes=np.min(prices,axis=0)
    return np.column_stack((mode,low,high,np.zeros(160,np.int64),directory,count)).astype(np.uint32),sizes


def _pages(n,plans,sizes,limit):
    """Each row belongs to exactly one output page under the native device grant."""
    row=0
    while row<160:
        end=row
        words=0
        while end<160 and words+sizes[end]<=limit:
            words+=int(sizes[end])
            end+=1
        if end==row:raise ValueError('A physical frame row exceeds the output grant.')
        if words:
            desc=plans[row:end].copy()
            desc[:,3]=np.cumsum(sizes[row:end],dtype=np.uint32)-sizes[row:end]
            yield row,np.r_[np.array([n,(n+127)//128,row],np.uint32),desc.ravel()],words
        row=end


def _restore(arrays,start,descriptor,raw):
    end=0
    for row,(mode,low,high,base,directory,count) in enumerate(descriptor[3:].reshape(-1,6),start):
        mode,low,high,base,directory,count=map(int,(mode,low,high,base,directory,count))
        if not mode:continue
        span=high-low
        directory_data=raw[base:base+directory]
        if mode==1:
            ids=slice(low,high)
            take=span
        else:
            if mode==2:
                bits=np.unpackbits(directory_data.astype('<u4',copy=False).view(np.uint8),bitorder='little')
                ids=np.flatnonzero(bits)+(low//32)*32
            else:ids=directory_data
            if len(ids)!=count or int(ids[0])!=low or int(ids[-1])+1!=high or np.any(ids[1:]<=ids[:-1]):
                raise ValueError('Invalid nondefault physical frame directory.')
            take=count
        data=base+directory
        end=data+2*take
        records=raw[data:end].reshape(take,2)
        present=(records[:,0]!=0)|(records[:,1]!=65535)
        if int(np.count_nonzero(present))!=count or not present[0] or not present[-1]:
            raise ValueError('Physical frame records disagree with their measured defaults.')
        for index,target in enumerate(arrays):
            value=records[:,index//2]
            target[row,ids]=value>>16 if index%2 else value
    if end!=len(raw):raise ValueError('Unconsumed physical frame payload.')


class FrameDefaults:
    def __init__(self,gpu):
        self.gpu=gpu
        self.web=getattr(gpu,'backend',None)=='webgpu'
        self.kernels=None
        self.maps=self.read_bytes=0
        self.distribution={}

    def capture(self):
        gpu=self.gpu
        self.maps=self.read_bytes=0
        self.distribution={}
        n=gpu.n
        if META_WORDS>=2*160*n:return None
        groups=(n+127)//128
        counts=np.array([n,groups,0],np.uint32)
        stages=('measure','scan','pack')
        if self.kernels is None:
            root=Path(__file__).resolve().parent
            if self.web:
                shader=gpu.gpu_device.create_shader_module(code=(root/'dfs_checkpoint_frame_defaults.wgsl').read_text(encoding='utf8'))
                self.kernels={s:gpu.gpu_device.create_compute_pipeline(layout='auto',compute={'module':shader,'entry_point':s}) for s in stages}
            else:
                module=gpu.cp.RawModule(code=(root/'dfs_checkpoint_frame_defaults.cu').read_text(encoding='utf8'))
                self.kernels={s:module.get_function(s) for s in stages}
        with ExitStack() as scope:
            if self.web:
                dev,usage=gpu.gpu_device,gpu.wgpu.BufferUsage
                limit=min(dev.limits['max-buffer-size'],dev.limits['max-storage-buffer-binding-size'])//4
                def buffer(resources,words,flags):
                    obj=dev.create_buffer(size=4*words,usage=flags);resources.callback(obj.destroy)
                    return obj
                def description(resources,data):
                    obj=buffer(resources,len(data),usage.STORAGE|usage.COPY_DST)
                    gpu.queue.write_buffer(obj,0,data)
                    return obj
                def dispatch(encoder,stage,x,y,pointers):
                    pipeline=self.kernels[stage]
                    group=dev.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[
                        {'binding':i,'resource':{'buffer':buf}} for i,buf in pointers])
                    compute=encoder.begin_compute_pass()
                    compute.set_pipeline(pipeline);compute.set_bind_group(0,group)
                    compute.dispatch_workgroups(x,y);compute.end()
                def read(handle,size):
                    gpu.readback_bytes+=4*size
                    return np.frombuffer(gpu.queue.read_buffer(handle,0,4*size),np.uint32)
                physical=list(enumerate(gpu.buffers[1:5]))
                work=buffer(scope,5*160*groups,usage.STORAGE)
                arg=description(scope,counts)
                head=buffer(scope,META_WORDS,usage.STORAGE|usage.COPY_SRC)
                cmd=dev.create_command_encoder()
                dispatch(cmd,'measure',groups,160,[*physical,(4,arg),(5,work)])
                dispatch(cmd,'scan',160,1,[(4,arg),(5,work),(6,head)])
                gpu.queue.submit([cmd.finish()])
                meta=read(head,META_WORDS)
                def fetch(plan,rows,size):
                    with ExitStack() as page:
                        arg=description(page,plan)
                        out=buffer(page,size,usage.STORAGE|usage.COPY_SRC)
                        cmd=dev.create_command_encoder()
                        dispatch(cmd,'pack',groups,rows,[*physical,(4,arg),(5,work),(6,out)])
                        gpu.queue.submit([cmd.finish()])
                        return read(out,size)
            else:
                cp=gpu.cp
                limit=2*160*n
                physical=tuple(getattr(gpu,f) for f in FIELDS)
                work=cp.empty(5*160*groups,cp.uint32)
                arg=cp.asarray(counts)
                head=cp.empty(META_WORDS,cp.uint32)
                self.kernels['measure']((groups,160),(128,),(*physical,arg,work))
                self.kernels['scan']((160,),(1,),(arg,work,head))
                meta=head.get()
                def fetch(plan,rows,size):
                    desc=cp.asarray(plan);out=cp.empty(size,cp.uint32)
                    self.kernels['pack']((groups,rows),(128,),(*physical,desc,work,out))
                    return out.get()
            self.maps=1
            self.read_bytes=4*META_WORDS
            plans,sizes=_prices(n,meta.reshape(160,3))
            self.distribution={mode:int((plans[:,0]==i).sum()) for i,mode in enumerate(('empty','span','bits','indices'))}
            if META_WORDS+int(sizes.sum())>=2*160*n:return None
            arrays=tuple(np.full((160,n),v,np.uint16) for v in DEFAULTS)
            for start,plan,size in _pages(n,plans,sizes,limit):
                raw=fetch(plan,(len(plan)-3)//6,size)
                self.maps+=1;self.read_bytes+=4*size
                _restore(arrays,start,plan,raw)
        return tuple(a.ravel() for a in arrays)
