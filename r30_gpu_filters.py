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
