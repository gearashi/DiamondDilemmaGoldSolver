"""Replay the Round 34 physical follow-up controls. Use --root-lifts for bounded Gold lifts."""
import argparse
from collections import Counter
import json

import numpy as np

from constraint_models import solve
from r27_incumbent import BoardModel
from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r29_gold_phase import SignedParity
from r31_sampling import ranked_start
from r32_fresh_geometry import skeleton_domains
from r32_fresh_phases import independent_phases, phase_cube, SignedRepair, lift_cube
from r32_fresh_skeleton import parity_seed
from r32_gold_bonds import inventory
from r34_contour_walk import contour_walk
from r34_geographic_mates import _post
from search_problem import SearchProblem
from test_r34_physical_followups import ring_puzzle


def phase_results(p):
    order=independent_phases(p)
    contraction={}
    for name,seq in (('numbered',range(1,p.n)),('independent',order)):
        q=SignedParity(p);q.pin(0,0)
        for a,b in sorted(inventory(p).required):q.bond(a,b)
        rank=len(q.base.pivots);counts={}
        for i,t in enumerate(seq):
            mark=len(q.base.trail)
            if not q.pin(t,0):
                q.base.rollback(mark)
                assert q.pin(t,1)
            if i+1 in (2,4,8,16,32,40):
                counts[i+1]=[len(q.base.pivots)-rank,len(q.base.forced())]
            if i+1==40:break
        contraction[name]=counts
    finite=ring_puzzle()
    fixed=[1<<(3*i) for i in range(finite.n)]
    decisions=[]
    for independent,width in ((False,finite.n-1),(True,len(independent_phases(finite)))):
        status=Counter()
        for i in range(1<<width):
            pins=phase_cube(finite,width,i,independent=independent)
            row=lift_cube(finite,inventory(finite).required,pins,fixed,2)
            status[row['status']]+=1
        decisions.append(dict(status))
    return dict(dimension=len(order),contraction=contraction,finite_lifts=decisions)


def mate_results(p):
    data={}
    for split in (False,True):
        q,ds=planted(1,5,34,split,True)
        q.domains=[[v for v in range(3*q.n) if d>>v&1] for d in ds]
        for engine in ('sat','cp-sat','cp-sat-circuit','cp'):
            groups=[]
            for enabled in (False,True):
                q.physical_gold_support=enabled
                row=solve(q,engine=engine,target='gold',seconds=10)
                groups.append(dict(status=row['status'],nodes=row['nodes']))
            data[str(split)+'_'+engine]=groups
    return dict(restrictions=_post(p,p.domains,lambda *args:None,lambda:None),controls=data)


def donor_results(p):
    m=BoardModel(p);ban=inventory(p).forbidden
    tally=Counter()
    for seed in range(8):
        rng=np.random.default_rng(seed)
        phases=rng.integers(0,2,p.n).tolist()
        b=SignedRepair(m,phases,ban)
        board=ranked_start(m,seed)
        for _ in range(16):
            cells=m.independent_set(board,rng)
            board=b.block(board,cells)
        tally.update(b.stats)
    return dict(counts=dict(tally),raw_vector_builds=3*tally['rows_computed'],
                template_bytes=sum(row.nbytes for row in b.geometry.templates.values()),
                final_phase_bytes=sum(row.nbytes for row in b.prices.values()))


def contour_results(p, full):
    def controls(q,limit,lifts):
        independent=[parity_seed(q,s,conditioned=True) for s in range(limit)]
        walk=contour_walk(q,0,limit=limit)
        sets=[[r['patterns'] for r in independent],walk['patterns']]
        data=dict(independent_cells=limit*q.n,walk_cells=walk['assignment_cells'],
                  unique=[len({tuple(row) for row in group}) for group in sets],
                  score=[max(r['actual_matches'] for r in independent),walk['best']['matched_edges']])
        if lifts:
            root=domain_bits(q)
            data['lifts']=[dict(Counter(lift_cube(q,inventory(q).required,{0:0},
                skeleton_domains(q,root,pat),2)['status'] for pat in group[:lifts])) for group in sets]
        return data
    return dict(root=controls(p,16,3 if full else 0),ring=controls(ring_puzzle(),6,6))


def check_restored_mate_prefixes(p, api):
    """Restore the three real non-loop forbidden mates above the prefix ownership floor."""
    from dfs_gpu import DFS, fill_order
    from gpu_reasoning import pack_frame
    import numpy as np
    if api == 'CUDA':
        factory, kwargs = DFS, {}
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        adapter = next(a for a in wgpu.gpu.enumerate_adapters_sync()
                       if a.info['backend_type'] == api and a.info['adapter_type'] == 'DiscreteGPU')
        factory, kwargs = WebGPUDFS, {'adapter': adapter}
    rules = inventory(p)
    pairs = sorted(rules.forbidden - rules.loops)
    order = fill_order(p.neighbors)
    a = int(order[0])
    results = {}
    for dynamic in (False, True):
        # Exercise a dynamic child whose static order rank differs from its actual rank.
        b = (next(int(cell) for cell in p.neighbors[a] if int(cell) != int(order[1]))
             if dynamic else int(order[1]))
        side = list(p.neighbors[a]).index(b)
        other_side = int(p.sides[a, side])
        rows = [[3*(y//3)+(side-y%3)%3, 3*(x//3)+(other_side-x%3)%3] for x, y in pairs]
        assert len({row[0] for row in rows}) == len(rows) > 0
        gpu = factory(p.masks, p.neighbors, p.sides, n=len(rows), order=order,
                      prefixes=[[head] for head, _ in rows], gold_data=p.tiles,
                      gpu_features={'dynamic_mrv': dynamic}, **kwargs)
        try:
            saved = {name: value.copy() for name, value in gpu.snapshot_checkpoint().items()}
            for lane, (head, child) in enumerate(rows):
                # Exact stack representation of a legacy search which consumed this child first.
                key = other_side*gpu.colors + int(gpu.host_reverse[head,side])
                first, end = map(int, gpu.host_single_offsets[key:key+2])
                shift, = np.flatnonzero(gpu.host_pool[first:end] == child)
                frame = pack_frame(first, int(shift), b) if dynamic else (first, int(shift))
                saved['firsts'][1,lane], saved['shifts'][1,lane] = frame
                saved['lengths'][1,lane] = end-first
                saved['cursors'][1,lane] = 1
                saved['boards'][b,lane] = child
                tile = child//3
                saved['used'][tile//32,lane] |= np.uint32(1 << (tile%32))
                saved['depths'][lane] = saved['maxdepths'][lane] = 2
            gpu.restore_checkpoint(saved)
            gpu.verify()
            gpu.step(1)
            gpu.verify()
            post = gpu.snapshot_checkpoint()
            for lane, (head, child) in enumerate(rows):
                assert int(post['states'][lane]) == 0 and int(post['depths'][lane]) == 1
                assert post['boards'][a,lane] == head and post['boards'][b,lane] == -1
                assert np.count_nonzero(post['boards'][:,lane] >= 0) == 1
            for field in ('prefix_codes','firsts','lengths','cursors','shifts','rng'):
                np.testing.assert_array_equal(post[field],saved[field])
            assert list(map(int, post['counters'].sum(axis=1))) == [0,0,len(rows)]
            results['dynamic' if dynamic else 'static'] = dict(lanes=len(rows), unplaced=len(rows),
                sibling_cursors_preserved=True, restored_active=len(rows), child_static_rank=int(np.where(order==b)[0][0]))
        finally:
            if hasattr(gpu,'close'):
                gpu.close()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--root-lifts',action='store_true')
    parser.add_argument('--gpu-prefix', choices=('CUDA','D3D12','Vulkan','Metal'))
    args=parser.parse_args()
    puzzle=SearchProblem(read_json(args.tiles))
    if args.gpu_prefix:
        print(json.dumps(check_restored_mate_prefixes(puzzle,args.gpu_prefix),indent=2))
        return
    print(json.dumps(dict(phases=phase_results(puzzle),mates=mate_results(puzzle),donors=donor_results(puzzle),
                          contours=contour_results(puzzle,args.root_lifts)),indent=2))


if __name__=='__main__':
    main()
