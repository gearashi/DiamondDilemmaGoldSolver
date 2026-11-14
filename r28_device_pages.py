"""Exact device-side dirty page detection for native CUDA checkpoints."""
import numpy as np


class DevicePages:
    """Cache the last exact host/device image. An interrupted capture is retried."""
    def __init__(self, words=4096):
        if type(words) is not int or words < 1:
            raise ValueError("Positive page size required")
        self.words = words
        self.host = self.device = None
        self.transferred = 0

    def capture(self, cp, state):
        if state.dtype != np.uint32 or not 0 < state.size <= np.iinfo(np.int32).max:
            raise ValueError("Native checkpoint pages require a nonempty uint32 state with a 32-bit index")
        # Ravel in physical memory order to avoid a second device-sized temporary.
        order = 'C' if state.flags.c_contiguous else 'F'
        flat = state.ravel(order=order)
        if self.device is None:
            mirror = flat.copy()
            host = mirror.get()
            self.device, self.host = mirror, host
            self.transferred += host.nbytes
        else:
            if flat.shape != self.device.shape or flat.dtype != self.device.dtype:
                raise ValueError("Checkpoint mirror shape changed")
            pages = (flat.size+self.words-1)//self.words
            flags = cp.empty(pages, dtype=cp.uint8)
            kernel = cp.RawKernel("""
            extern "C" __global__ void changed(const unsigned *a, const unsigned *b,
                        unsigned char *flags, int n, int page) {
                const int start=blockIdx.x*page;
                int different=0;
                for(int j=start+threadIdx.x;j<min(start+page,n);j+=blockDim.x)
                    different |= (a[j]!=b[j]);
                int any=__syncthreads_or(different);
                if(threadIdx.x==0) flags[blockIdx.x]=any;
            }""", 'changed')
            kernel((pages,), (128,), (flat, self.device, flags, np.int32(flat.size), np.int32(self.words)))
            dirty = np.flatnonzero(flags.get())
            self.transferred += flags.nbytes
            if dirty.size:
                # Preserve the last host snapshot exposed to a writer.
                new_host = self.host.copy()
                groups = np.split(dirty, np.flatnonzero(np.diff(dirty)!=1)+1)
                for group in groups:
                    sl = slice(int(group[0])*self.words, min((int(group[-1])+1)*self.words, flat.size))
                    new_host[sl] = flat[sl].get()
                    self.transferred += new_host[sl].nbytes
                try:
                    for group in groups:
                        sl = slice(int(group[0])*self.words, min((int(group[-1])+1)*self.words, flat.size))
                        self.device[sl] = flat[sl]
                    cp.cuda.get_current_stream().synchronize()
                except BaseException:
                    self.host = self.device = None
                    raise
                self.host = new_host
        result = np.ascontiguousarray(self.host.reshape(state.shape, order=order)).view()
        result.setflags(write=False)
        return result
