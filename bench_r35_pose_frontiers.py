"""Compare authenticated original-pose conflicts with unrestricted CPU/CUDA Gold domains."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random

from constraint_symmetry import proper_rotations
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


class RotatedFacts:
    def __init__(self, problem):
        self.record = active_catalog(problem)
        self.images = []
        self.rows_by_cell = defaultdict(list)
        if self.record is None:
            return
        self.rotations = proper_rotations(problem.board)
        seed, rows = self.record['codes'], self.record['rows']
        self.masks = tuple(sum(1 << c for c in row) for row in rows)
        for i, row in enumerate(rows):
            for c in row:
                self.rows_by_cell[c].append(i)
        for rotation in self.rotations:
            # Each mask is indexed in original reference coordinates.
            image = rotation.apply(seed)
            self.images.append([(rotation.cells[c], 1 << image[rotation.cells[c]]) for c in range(problem.n)])

    def forced(self, domains):
        removed = set()
        contradiction = False
        for pairs in self.images:
            held = 0
            possible = 0
            candidates = set()
            for c, (site, bit) in enumerate(pairs):
                if domains[site] & bit:
                    possible |= 1 << c
                    if domains[site] == bit:
                        held |= 1 << c
                        candidates.update(self.rows_by_cell[c])
            for i in candidates:
                row = self.masks[i]
                if row & ~possible:
                    continue
                left = row & ~held
                if not left:
                    contradiction = True
                elif left & (left-1) == 0:
                    removed.add(pairs[left.bit_length()-1])
            if held.bit_count() >= len(pairs)-self.record['lower_bound']:
                if held.bit_count() > len(pairs)-self.record['lower_bound']:
                    contradiction = True
                else:
                    removed.update(pairs[c] for c in range(len(pairs)) if (possible & ~held) >> c & 1)
        return contradiction, sorted(removed)


class Compiled:
    def __init__(self, problem, directory):
        from constraint_dfs import _build_support_tables
        from constraint_native_search import NativeSearch
        self.engine = NativeSearch(problem.n, directory=directory, max_depth=192)
        full = (1 << (3*problem.n))-1
        tables = _build_support_tables(problem, [full]*problem.n, 'gold', lambda:None)
        self.engine.configure_propagation(tables)

    def close(self):
        self.engine.close()

    def narrow(self, domains):
        outcome = []
        def observe(ds, decisions):
            assert self.engine.basic_propagated
            outcome.append(ds)
            return {'action':'reject'}
        self.engine.run([dict(domains=domains, cell=-1, choices=[], next=0)], observe)
        return outcome[0] if outcome else None


def anchored(full, poses):
    ds = list(full)
    for c, v in poses:
        ds[c] = 1 << v
    return ds


def saved_stacks(checkpoint_dir):
    for name in ('dfs-gold', 'hybrid-gold'):
        path = checkpoint_dir/name/'checkpoint.json'
        if not path.is_file():
            continue
        data = json.loads(path.read_bytes())['payload']
        data = data.get('dfs',data)
        if data['fingerprint'] != 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
            continue
        for depth, frame in enumerate(data['stack']):
            yield name+'-'+str(depth), [int(v,16) for v in frame['domains']]


def observe(count, winners, label, ds, facts):
    count['cubes'] += 1
    if ds is None:
        count['baseline_failed'] += 1
        return
    count['baseline_live'] += 1
    count['baseline_live_values'] += sum(d.bit_count() for d in ds)
    conflict, deleted = facts.forced(ds)
    if deleted or conflict:
        count['improved'] += 1
        count['extra_failures'] += int(conflict)
        count['distinct_value_units'] += len(deleted)
        if len(winners) < 50:
            winners.append(dict(label=label,conflict=conflict,units=[(c,b.bit_length()-1) for c,b in deleted]))


def cpu_screen(problem,facts,args):
    from time import perf_counter
    start = perf_counter()
    full = [(1 << (3*problem.n))-1]*problem.n
    original = facts.record['codes']
    cpu = Compiled(problem,args.native)
    result = {}
    try:
        groups = {
            'fresh':[('root',full)],
            'historical':list(saved_stacks(args.checkpoints)),
            'singletons':[(str(c),anchored(full,[(c,v)])) for c,v in enumerate(original)],
        }
        rng=random.Random(89)
        # This includes every distinct one-pose-short core in the original
        # orientation. Other orientations are equivalent on the full open mesh.
        subs = sorted({tuple(row[:i]+row[i+1:]) for row in facts.record['rows'] for i in range(len(row)) if len(row)>2})
        rng.shuffle(subs)
        if args.subsets:
            subs=subs[:args.subsets]
        groups['near_unit']=[(str(list(s)),anchored(full,[(c,original[c]) for c in s])) for s in subs]
        for name,cases in groups.items():
            count=Counter();winners=[]
            for label,ds in cases:
                observe(count,winners,label,cpu.narrow(ds),facts)
            result[name]={'counts':dict(count),'examples':winners}
            print(name,dict(count),round(perf_counter()-start,2),flush=True)
        post=facts.record['post_master']
        cases=sorted({tuple(row) for row in post} | {tuple(row[:i]+row[i+1:]) for row in post for i in range(len(row))})
        from r35_pose_departures import propagate,indexed_rows
        post_index=indexed_rows(tuple(tuple(row) for row in post))
        scope=Counter()
        for cells in cases:
            ds=cpu.narrow(anchored(full,[(c,original[c]) for c in cells]))
            scope['tested']+=1
            if ds is None:
                scope['native_closed']+=1
                continue
            base_conflict,base_units=facts.forced(ds)
            if base_conflict:
                scope['base_already_closed']+=1
                continue
            reduced,report=propagate(ds,facts.images,post_index,160)
            if report['over']:
                scope['extra_conflict']+=1
                continue
            base_units=set(base_units)
            extra=sum((c,1<<original[c]) not in base_units and bool(old & (1<<original[c]) & ~new)
                      for c,(old,new) in enumerate(zip(ds,reduced)))
            scope['extra_unit_cubes']+=int(extra>0)
            scope['extra_units']+=extra
        result['post_master']=dict(scope)
        # Real paths produced by the compiled backend with a deterministic MRV
        # split. Every node callback has already reached its full-width seam /
        # owner fixed point. Its actual branches and child visits are counted.
        choices_runs=[]
        for seed in args.seeds:
            rng=random.Random(seed)
            rank=list(range(problem.n));rng.shuffle(rank)
            codes=list(range(3*problem.n));rng.shuffle(codes)
            count=Counter();winners=[]
            def choose(ds, decisions):
                label=str(len(decisions))+'-'+str(count['cubes'])
                observe(count,winners,label,ds,facts)
                c=min((i for i in range(problem.n) if ds[i].bit_count()>1),key=lambda i:(ds[i].bit_count(),rank[i]),default=None)
                if c is None:
                    return {'action':'reject'}
                return dict(action='branch',cell=c,choices=[1<<v for v in codes if ds[c]>>v&1])
            out=cpu.engine.run([dict(domains=full,cell=-1,choices=[],next=0)],choose,max_steps=args.steps)
            choices_runs.append(dict(seed=seed,counts=dict(count),examples=winners,stats=out['stats']))
        result['fresh_paths']=choices_runs
    finally:
        cpu.close()
    result['seconds']=perf_counter()-start
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('../data/tiles.json'))
    parser.add_argument('--native',type=Path,default=Path('../.native'))
    parser.add_argument('--checkpoints',type=Path,default=Path('../runtime/searches'))
    parser.add_argument('--subsets',type=int,default=0)
    parser.add_argument('--steps',type=int,default=1024)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,17,89])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()),physical_gold_support=True)
    f=RotatedFacts(p)
    corpus=set()
    for image in f.images:
        for row in f.record['rows']:
            corpus.add(tuple(sorted((image[c][0],image[c][1].bit_length()-1) for c in row)))
    out=dict(fingerprint=p.fingerprint,rows=len(f.record['rows']),distinct_images=len(corpus),
             widths=dict(Counter(map(len,f.record['rows']))),cpu=cpu_screen(p,f,args))
    args.output.write_text(json.dumps(out,indent=2),encoding='utf8')

if __name__=='__main__':main()
