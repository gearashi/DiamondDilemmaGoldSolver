"""Batched CPU and CUDA filters for support geometry and physical movement cores."""
import itertools
import numpy as np


def core_passes(cells, core_masks, n, backend='cpu'):
    """Return one boolean per center/support; true means every required set is hit."""
    cells = np.asarray(cells,dtype=np.int32)
    if cells.ndim!=2 or np.any(cells<0) or np.any(cells>=n):
        raise ValueError("Invalid support matrix.")
    if backend not in ('cpu','cuda'):
        raise ValueError("Unknown filter backend.")
    xp = np
    if backend=='cuda':
        import cupy as xp
    indices = xp.asarray(cells)
    output = []
    for masks in core_masks:
        if not masks:
            output.append(xp.ones(len(cells),dtype=bool))
            continue
        flags = np.array([[bool(core & (1<<i)) for i in range(n)] for core in masks],dtype=bool)
        rows = xp.asarray(flags)
        output.append(xp.all(xp.any(rows[:,indices],axis=-1),axis=0))
    result = xp.stack(output)
    return xp.asnumpy(result) if backend=='cuda' else result


def contact_counts(cells, close, backend='cpu'):
    """Count distinct distance-at-most-two pairs, an upper bound on cross-cycle pairs."""
    cells = np.asarray(cells,dtype=np.int32)
    close = np.asarray(close,dtype=bool)
    if (cells.ndim!=2 or close.ndim!=2 or close.shape[0]!=close.shape[1]
            or np.any(cells<0) or np.any(cells>=len(close)) or backend not in ('cpu','cuda')):
        raise ValueError('Invalid contact query.')
    pairs = list(itertools.combinations(range(cells.shape[1]),2))
    if not pairs:
        return np.zeros(len(cells),dtype=np.int32)
    xp = np
    if backend=='cuda':
        import cupy as xp
    left,right = np.asarray(pairs).T
    ids = xp.asarray(cells)
    graph = xp.asarray(close)
    result = graph[ids[:,left],ids[:,right]].sum(axis=1)
    return xp.asnumpy(result) if backend=='cuda' else result


class CudaPageFilters:
    """Keep a route's immutable topology and core rows on-device across ranked pages."""
    def __init__(self, n, core_masks, close):
        import cupy as cp
        self.n = n
        self.has_contacts = close is not None
        self.has_cores = core_masks is not None
        offsets = [0]
        masks = []
        for center in core_masks or ():
            masks.extend(center)
            offsets.append(len(masks))
        self.centers = len(offsets) - 1
        flags = np.array([[bool(int(mask) & (1 << c)) for c in range(n)] for mask in masks],dtype=np.uint8)
        graph = np.asarray(close,dtype=np.uint8) if self.has_contacts else np.empty(0,dtype=np.uint8)
        self.graph = cp.asarray(np.ascontiguousarray(graph))
        self.flags = cp.asarray(flags)
        self.offsets = cp.asarray(offsets,dtype=cp.int32)
        self.kernel = cp.RawKernel(r"""
        extern "C" __global__ void filter_page(
            const int *cells, int count, int width, int n, const unsigned char *graph,
            const unsigned char *flags, const int *offsets, int centers, int contacts,
            unsigned char *out) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= count) return;
            const int *support = cells + row * width;
            if (contacts) {
                int total = 0;
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < i; ++j)
                        total += graph[support[j] * n + support[i]];
                out[row] = total;
            }
            for (int c = 0; c < centers; ++c) {
                bool valid = true;
                for (int t = offsets[c]; t < offsets[c + 1] && valid; ++t) {
                    valid = false;
                    for (int i = 0; i < width; ++i)
                        if (flags[t * n + support[i]]) { valid = true; break; }
                }
                out[(c + contacts) * count + row] = valid;
            }
        }""", "filter_page")

    def evaluate(self, cells):
        import cupy as cp
        cells = np.ascontiguousarray(cells,dtype=np.int32)
        if (cells.ndim != 2 or cells.shape[1] > 23 or np.any(cells < 0) or np.any(cells >= self.n)):
            raise ValueError("Invalid ranked support for packed CUDA filters.")
        count, width = cells.shape
        device_cells = cp.asarray(cells)
        contacts = int(self.has_contacts)
        out = cp.empty((contacts + self.centers, count), dtype=cp.uint8)
        if count:
            self.kernel(((count + 127) // 128,), (128,),
                        (device_cells, np.int32(count), np.int32(width), np.int32(self.n), self.graph,
                         self.flags, self.offsets, np.int32(self.centers), np.int32(contacts), out))
        values = cp.asnumpy(out)
        return (values[contacts:].view(np.bool_) if self.has_cores else None,
                values[0] if self.has_contacts else None)
