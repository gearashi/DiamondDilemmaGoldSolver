"""Recover full physical frame arrays from their checked nondefault values on the device."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np

FIELDS=('firsts','lengths','cursors','shifts')
DEFAULTS=(0,0,65535,0)
META_WORDS=160*7


def _prices(n,meta):
    """Price raw row spans, row bitmaps and indexed columns with lossless field ranges."""
    meta=np.asarray(meta)
    if meta.shape!=(160,7):raise ValueError('Invalid physical frame summary size.')
    count,low,high=meta[:,:3].astype(np.int64).T
    minimum=(meta[:,3:]&65535).astype(np.int64)
    maximum=(meta[:,3:]>>16).astype(np.int64)
    empty=count==0
    if (np.any(count<0) or np.any(empty&((low!=n)|(high!=0))) or
            np.any((~empty)&((low<0)|(high>n)|(low>=high)|(count>high-low))) or
            np.any((minimum>maximum)&(~empty[:,None]))):
        raise ValueError('Inconsistent physical frame summary.')
    minimum[empty]=maximum[empty]=0
    span=np.where(empty,0,high-low)
    mask=np.where(empty,0,(high-1)//32-low//32+1)
    holes=count<span
    span_min=np.where(holes[:,None],np.minimum(minimum,DEFAULTS),minimum)
    span_max=np.where(holes[:,None],np.maximum(maximum,DEFAULTS),maximum)
    widths=np.frexp(maximum-minimum)[1].astype(np.int64)
    span_widths=np.frexp(span_max-span_min)[1].astype(np.int64)
    packed=(count*widths.sum(axis=1)+31)//32
    prices=np.stack(((span*span_widths.sum(axis=1)+31)//32,mask+packed,count+packed))
    mode=np.where(empty,0,1+np.argmin(prices,axis=0))
    directory=np.where(mode==2,mask,np.where(mode==3,count,0))
    bias=np.where((mode==1)[:,None],span_min,minimum)
    widths=np.where((mode==1)[:,None],span_widths,widths)
    plans=np.column_stack((mode,low,high,np.zeros(160,np.int64),directory,count,
                bias[:,0]|(bias[:,1]<<16),bias[:,2]|(bias[:,3]<<16),
                (widths << np.arange(0,20,5)).sum(axis=1),widths.sum(axis=1)))
    return plans.astype(np.uint32),np.min(prices,axis=0)


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
        if np.any(plans[row:end,0]):
            desc=plans[row:end].copy()
            desc[:,3]=np.cumsum(sizes[row:end],dtype=np.uint32)-sizes[row:end]
            yield row,np.r_[np.array([n,(n+127)//128,row],np.uint32),desc.ravel()],words
        row=end


def _restore(arrays,start,descriptor,raw):
    end=0
    for row,plan in enumerate(descriptor[3:].reshape(-1,10),start):
        mode,low,high,base,directory,count,a,b,widths,total=map(int,plan)
        if not mode:continue
        directory_data=raw[base:base+directory]
        if mode==1:
            ids=slice(low,high);take=high-low
        else:
            if mode==2:
                bits=np.unpackbits(directory_data.astype('<u4',copy=False).view(np.uint8),bitorder='little')
                ids=np.flatnonzero(bits)+(low//32)*32
            else:ids=directory_data
            if len(ids)!=count or int(ids[0])!=low or int(ids[-1])+1!=high or np.any(ids[1:]<=ids[:-1]):
                raise ValueError('Invalid nondefault physical frame directory.')
            take=count
        words=(total*take+31)//32
        end=base+directory+words
        if end>len(raw):raise ValueError('Truncated physical frame data.')
        stream=np.zeros(words+1,np.uint32)
        stream[:words]=raw[base+directory:end]
        offsets=np.arange(take,dtype=np.uint32)*total
        present=np.zeros(take,bool)
        for index,target in enumerate(arrays):
            width=(widths>>(5*index))&31
            bias=((a if index<2 else b)>>(16*(index%2)))&65535
            value=bias
            if width:
                word=offsets>>5
                pair=stream[word].astype(np.uint64)|(stream[word+1].astype(np.uint64)<<32)
                value=(((pair>>(offsets&31))&((1<<width)-1)).astype(np.uint32)+bias)
                if np.any(value>65535):raise ValueError('Packed physical field exceeds uint16.')
                offsets+=width
            target[row,ids]=value
            present|=value!=DEFAULTS[index]
        if int(np.count_nonzero(present))!=count or not present[0] or not present[-1]:
            raise ValueError('Physical frame records disagree with their measured defaults.')
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
        if 5+META_WORDS>=2*160*n:return None
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
                def read(handle,size,offset=0):
                    gpu.readback_bytes+=4*size
                    return np.frombuffer(gpu.queue.read_buffer(handle,4*offset,4*size),np.uint32)
                physical=list(enumerate(gpu.buffers[1:5]))
                work=buffer(scope,9*160*groups,usage.STORAGE)
                arg=description(scope,counts)
                head=buffer(scope,5+META_WORDS,usage.STORAGE|usage.COPY_SRC)
                cmd=dev.create_command_encoder()
                dispatch(cmd,'measure',groups,160,[*physical,(4,arg),(5,work)])
                dispatch(cmd,'scan',160,1,[(4,arg),(5,work),(6,head)])
                gpu.queue.submit([cmd.finish()])
                active=read(head,5)
                def metadata():return read(head,META_WORDS,5)
                def fetch(plan,rows,size):
                    with ExitStack() as page:
                        arg=description(page,plan)
                        out=buffer(page,1+size,usage.STORAGE|usage.COPY_SRC)
                        cmd=dev.create_command_encoder()
                        dispatch(cmd,'pack',groups,rows,[*physical,(4,arg),(5,work),(6,out)])
                        gpu.queue.submit([cmd.finish()])
                        return read(out,1+size)
            else:
                cp=gpu.cp
                limit=2*160*n
                physical=tuple(getattr(gpu,f) for f in FIELDS)
                work=cp.empty(9*160*groups,cp.uint32)
                arg=cp.asarray(counts)
                head=cp.zeros(5+META_WORDS,cp.uint32)
                self.kernels['measure']((groups,160),(128,),(*physical,arg,work))
                self.kernels['scan']((160,),(1,),(arg,work,head))
                active=head[:5].get()
                def metadata():return head[5:].get()
                def fetch(plan,rows,size):
                    desc=cp.asarray(plan);out=cp.zeros(1+size,cp.uint32)
                    self.kernels['pack']((groups,rows),(128,),(*physical,desc,work,out))
                    return out.get()
            self.maps=1
            self.read_bytes=20
            def empty_frames():return tuple(np.full((160,n),v,np.uint16) for v in DEFAULTS)
            if not active.any():
                self.distribution={'empty':160}
                return tuple(a.ravel() for a in empty_frames())
            meta=metadata().reshape(160,7)
            self.maps+=1;self.read_bytes+=4*META_WORDS
            advertised=np.unpackbits(active.astype('<u4',copy=False).view(np.uint8),bitorder='little')
            if not np.array_equal(meta[:,0]!=0,advertised):raise ValueError('Conflicting physical frame activity.')
            plans,sizes=_prices(n,meta)
            self.distribution={mode:int((plans[:,0]==i).sum()) for i,mode in enumerate(('empty','span','bits','indices'))}
            pages=list(_pages(n,plans,sizes,limit-1))
            if self.read_bytes+4*sum(size+bool(size) for _,_,size in pages)>=8*160*n:return None
            arrays=empty_frames()
            for start,plan,size in pages:
                raw=np.empty(0,np.uint32)
                if size:
                    packet=fetch(plan,(len(plan)-3)//10,size)
                    self.maps+=1;self.read_bytes+=4*len(packet)
                    if packet[0]:raise ValueError('Physical fields changed during frame packing.')
                    raw=packet[1:]
                _restore(arrays,start,plan,raw)
        return tuple(a.ravel() for a in arrays)
