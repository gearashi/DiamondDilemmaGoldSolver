"""CUDA accelerator for complete small exchange shells. Host validates all survivors."""
import math
import numpy as np

# For nonoverlapping stars, individual substitutions add exactly. When stars
# overlap, recompute their union including every simultaneous substitution.
SOURCE = r'''
extern "C" __global__ void shell(
  long long start, int count, int n, int k, const long long *choose, const int *perm,
  const int *nb, const int *base, const unsigned char *masks, const int *delta,
  const unsigned char *overlap, int total, int limit, unsigned char *out) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count) return;
    long long ix=start+tid;
    int p[6],left=n;
    for(int j=k;j>0;--j) {
        int lo=j-1, hi=left-1;
        while(lo<hi) {
            int mid=(lo+hi+1)/2;
            if(choose[mid*(k+1)+j]<=ix) lo=mid;
            else hi=mid-1;
        }
        p[j-1]=lo; ix-=choose[lo*(k+1)+j]; left=lo;
    }
    int bound=total;
    bool close=false;
    for(int j=0;j<k;++j) {
        bound+=delta[p[j]*n+p[perm[j]]];
        for(int l=0;l<j;++l) close |= (overlap[p[j]*n+p[l]]!=0);
    }
    if(close) {
        bound=total;
        int seen[24], used=0;
        for(int j=0;j<k;++j) for(int s=-1;s<3;++s) {
            int c=s<0 ? p[j] : nb[3*p[j]+s];
            if(c==n) continue;
            bool dup=false;
            for(int a=0;a<used;++a) if(seen[a]==c) dup=true;
            if(dup) continue;
            seen[used++]=c;
            int source=c, neigh[3]={nb[c*3],nb[c*3+1],nb[c*3+2]};
            for(int a=0;a<k;++a) {
                if(c==p[a]) source=p[perm[a]];
                for(int b=0;b<3;++b) if(nb[c*3+b]==p[a]) neigh[b]=p[perm[a]];
            }
            int best=0;
            for(int r=0;r<3;++r) {
                int v=0;
                for(int b=0;b<3;++b) v+=masks[(((source*3+r)*3+b)*(n+1))+neigh[b]];
                if(v>best) best=v;
            }
            bound+=best-base[c];
        }
    }
    out[tid]=bound>=limit;
}
'''


def unrank(rank, n, size):
    cells = []
    for width in range(size, 0, -1):
        lo, hi = width-1, n-1
        while lo < hi:
            mid = (lo+hi+1)//2
            if math.comb(mid, width) <= rank:
                lo = mid
            else:
                hi = mid-1
        cells.append(lo)
        rank -= math.comb(lo, width)
        n = lo
    return cells[::-1]


class CudaShell:
    def __init__(self, bound, size):
        import cupy as cp
        if not 2 <= size <= 6 or bound.model.problem.fixed:
            raise ValueError('CUDA shell supports distances 2 to 6 with no fixed cells; use CPU otherwise.')
        n = bound.n
        coefficients = np.zeros((n, n), dtype=np.int32)
        neighborhoods = [set([c, *bound.neighbors[c]]) - {n} for c in range(n)]
        for cell in range(n):
            centers = np.asarray(sorted(neighborhoods[cell]))
            nb = bound.neighbors[centers]
            replacement = np.arange(n)[:, None]
            central = np.where(centers == cell, bound.pieces[replacement], bound.pieces[centers])
            others = np.where(nb == cell, bound.pieces[replacement[..., None]], bound.pieces[nb])
            coefficients[cell] = (bound.stars(central, others)-bound.base[centers]).sum(axis=1)
        overlap = np.array([[not a.isdisjoint(b) for b in neighborhoods] for a in neighborhoods], dtype=np.uint8)
        masks = bound.support[bound.masks[bound.tiles][..., None], bound.pieces]
        choose = np.asarray([[math.comb(i, j) for j in range(size+1)] for i in range(n+1)], dtype=np.int64)
        self.args = tuple(map(cp.asarray, (bound.neighbors[:n].astype(np.int32), bound.base.astype(np.int32),
                                           masks.astype(np.uint8), coefficients, overlap)))
        self.choose = cp.asarray(choose)
        self.fn = cp.RawKernel(SOURCE, 'shell')
        self.cp, self.n, self.size, self.total = cp, n, size, bound.total

    def survivors(self, offset, count, order, target):
        cp = self.cp
        out = cp.empty(count, dtype=cp.uint8)
        i = np.int32
        self.fn(((count+127)//128,), (128,), (np.int64(offset), i(count), i(self.n), i(self.size),
                self.choose, cp.asarray(order, dtype=cp.int32), *self.args, i(self.total), i(2*target), out))
        return [unrank(offset+int(index), self.n, self.size) for index in cp.asnumpy(cp.flatnonzero(out))]
