"""Compare scoped Round 27 reasoning against exact finite oracles and released engines."""
from collections import Counter
from itertools import product
import argparse
import json
from pathlib import Path
import random
import time

from r27_reason import algebra, concurrent_sat, cost_bound, maximize_seams, orbit_choices, scoped_search
from r27_reason_cost import SeamCostCache
from search_problem import SearchProblem
from test_r27_reason import answers, corpus
from test_constraint_models import pair_problem
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_separator import ring
from r27_reason_search import ScopedSearch
from test_constraint_dfs import board


def odd_cycles():
    # Each seam requires opposite endpoint colors. Both isolated odd cycles have
    # a fractional half/half solution but no integral two-coloring.
    p = SearchProblem([{'id':i,'segments':[[[0,v],[1,v]]]} for i,v in enumerate([2]*3+[10]*3)],
                      board(6,[(i,1,3*(i//3)+(i+1)%3,0) for i in range(6)]))
    return p,[[0,9],[3,12],[6,15]]*2


def finite():
    from constraint_dfs import solve as released_dfs
    from constraint_models import solve as released_sat
    report = {'released':{},'split':{},'dynamic':{},'cost':{},'algebra':{},'orbits':{},'parallel':{}}
    tetra = SearchProblem(*tetrahedron_tiles())
    fixtures = {'pair':pair_problem(),'pair-multiple-loops':pair_problem(multiple_loops=True),'tetra':tetra,
                'ring4':ring(4),'ring6':ring(6)}
    for label,p in fixtures.items():
        report['released'][label] = {}
        for engine,f in [('dfs',released_dfs),('sat',released_sat)]:
            result = f(p,seconds=5)
            report['released'][label][engine] = {k:result[k] for k in ('status','nodes','complete')}
        for pilots in (0,3):
            result = concurrent_sat(p,pilots=pilots)
            report['parallel'][label+'_'+str(pilots)] = {k:v for k,v in result.items() if k!='codes'}
        orbit = orbit_choices(p)
        report['orbits'][label] = {'before':len(p.domains[orbit['cell']]),'after':len(orbit['choices']),
                                   'status':orbit['status'],'work':orbit['stats'].get('isomorphic_work_ticks',0)}
    training = {}
    training_work = Counter()
    predictor = {}
    for group,seed in [('calibration',271609),('heldout',271611)]:
        rng = random.Random(seed)
        search = {m:Counter() for m in ('mrv','strong','backdoor','backjump','dynamic')}
        if group=='heldout':
            predictor = {key:sum(vals)/len(vals) for key,vals in training.items()}
            search['learned'] = Counter()
        best = {}
        reliability = Counter()
        for label in ('pair','pair-multiple-loops','tetra','ring4','ring6'):
            p = fixtures[label]
            for rows in corpus(p,rng,14,.56):
                boards = answers(p,rows)
                valid = {b for b,_,gold in boards if gold}
                results = {}
                if group=='calibration':
                    tutor = ScopedSearch(p,rows,'gold',100000,time.monotonic()+5)
                    live = tutor.reduced(rows)
                    if live:
                        active = sorted((i for i,r in enumerate(live) if len(r)>1),key=lambda c:(len(live[c]),c))[:3]
                        for c in active:
                            features = tutor.split_features(live,c)
                            children = [tutor.reduced(r) for r in tutor.split(live,c)]
                            size = max((sum(len(row)-1 for row in r) if r else -1) for r in children)
                            training.setdefault(features,[]).append(size/max(1,sum(len(row)-1 for row in live)))
                            tutor.stats['training_children'] += 2
                    training_work.update(tutor.stats)
                for m,sums in search.items():
                    r = scoped_search(p,domains=rows,mode=m,predictor=predictor)
                    if r['status'] == 'unknown' or bool(r['codes']) != bool(valid) or r['codes'] and tuple(r['codes']) not in valid:
                        raise RuntimeError('Scoped solver failed exhaustive comparison.')
                    sums.update(r['stats'])
                    results[m] = r['stats']
                if 'learned' in results:
                    for m in ('strong','learned'):
                        baseline = results['mrv']['nodes']
                        nodes = results[m]['nodes']
                        reliability[m+('_win' if nodes<baseline else '_loss' if nodes>baseline else '_tie')] += 1
                for name,base,cand in [('dynamic','backjump','dynamic'),('strong','mrv','strong'),('backdoor','mrv','backdoor')]:
                    metric = 'assignments' if name=='dynamic' else 'nodes'
                    benefit = results[base].get(metric,0)-results[cand].get(metric,0)
                    if benefit > best.get(name,{}).get('gain',0):
                        best[name] = {'case':label,'domains':rows,'gain':benefit,'base':results[base],'candidate':results[cand]}
        report['split'][group] = {'seed':seed,'scopes':70,'totals':{m:dict(v) for m,v in search.items()},'positive_witness':best,'heldout_reliability':dict(reliability)}
    report['predictor_training'] = {'features':len(training),'stats':dict(training_work)}
    p = tetra
    rng = random.Random(270017)
    vac_gain = shift_gain = 0
    caches = [SeamCostCache(p,limit=0),SeamCostCache(p)]
    cache_work = [Counter(),Counter()]
    for domains in corpus(p,rng,54,.35):
        if not all(domains):
            continue
        boards = answers(p,domains)
        if not boards:
            continue
        lower = len(p.edges)-max(score for _,score,_ in boards)
        r = cost_bound(p,domains=domains)
        if not 0<=r['lower_bound']<=lower:
            raise RuntimeError('Invalid defect lower bound.')
        vac_gain += r['vac_bound']>r['shift_bound']
        shift_gain += r['shift_bound']>r['vac_bound']
        for i,cache in enumerate(caches):
            for rows in (domains,[domains[0][:1],*domains[1:]],domains):
                cache_work[i].update(cost_bound(p,domains=rows,cache=cache)['stats'])
    report['cost'] = {'vac_stronger':vac_gain,'shift_stronger':shift_gain,
                      'cold':dict(cache_work[0]),'incremental':dict(cache_work[1])}
    p = pair_problem()
    rng = random.Random(270021)
    for rows in corpus(p,rng,65,.45):
        if not all(rows):
            continue
        valid = any(s==len(p.edges) for _,s,_ in answers(p,rows))
        for method in ('lp','gf2'):
            degrees = [algebra(p,domains=rows,method=method,degree=d) for d in (1,2)]
            if valid and any(r['status']=='certified_infeasible' for r in degrees):
                raise RuntimeError('Invalid algebraic exclusion.')
            tag = str(tuple(r['status']=='certified_infeasible' for r in degrees))
            result = report['algebra'].setdefault(method,{'counts':Counter()})
            result['counts'][tag] += 1
            if tag=='(False, True)' and 'additional' not in result:
                result['additional']={'rows':rows,'generated':[v['rows'] for v in degrees]}
    for label in ('pair','pair-multiple-loops','ring4','ring6'):
        p = fixtures[label]
        for mode in ('mrv','orbit'):
            run = scoped_search(p,mode=mode,max_checks=500000)
            report['orbits'][label][mode] = {'status':run['status'],'stats':run['stats']}
    p,rows = odd_cycles()
    if any(s==len(p.edges) for _,s,_ in answers(p,rows)):
        raise RuntimeError('Odd-cycle calibration fixture unexpectedly feasible.')
    report['algebra']['odd_cycles'] = {method:[{k:v for k,v in algebra(p,domains=rows,method=method,degree=d).items() if k!='certificate'}
                for d in (1,2)] for method in ('lp','gf2')}
    return report


def original(tiles,seed_path,checkpoint=None,seconds=30):
    p = SearchProblem(json.loads(tiles.read_text(encoding='utf-8-sig')))
    b = json.loads(seed_path.read_text(encoding='utf-8-sig'))['codes']
    result = {'seed_score':p.validate(b)['matched_edges'],'scope':p.fingerprint}
    maxsat = maximize_seams(p,seed_board=b,threshold=min(len(p.edges),result['seed_score']+1),seconds=seconds,conflicts=25000)
    result['threshold_test'] = {k:v for k,v in maxsat.items() if k!='codes'}
    baseline = cost_bound(p,probes=2,shift_pairs=25000)
    result['root_cost'] = {k:v for k,v in baseline.items() if k!='certificate'}
    if checkpoint:
        stack = json.loads(checkpoint.read_text(encoding='utf-8-sig'))['payload']['stack']
        screens = []
        for pos in sorted({0,len(stack)//2,len(stack)-1}):
            rows = [[i for i in range(3*p.n) if int(mask,16)>>i&1] for mask in stack[pos]['domains']]
            record = {'stack_depth':pos,'active':sum(len(r)>1 for r in rows),'placement_variables':sum(map(len,rows)),
                      'orbits':orbit_choices(p,domains=rows)['status'],'gf2':algebra(p,domains=rows)['status'],
                      'lp':algebra(p,domains=rows,method='lp')['status']}
            bound = cost_bound(p,domains=rows,probes=3,shift_pairs=25000)
            record['defect_bound'] = bound['lower_bound']
            record['defect_work'] = bound['stats']
            for mode in ('backdoor','dynamic','strong','mrv','backjump'):
                r = scoped_search(p,domains=rows,mode=mode,max_checks=10000)
                record[mode] = {'status':r['status'],'stats':r['stats']}
            screens.append(record)
        result['checkpoint_scopes'] = screens
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path)
    parser.add_argument('--seed-board',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if bool(args.tiles) != bool(args.seed_board):
        parser.error('Provide tiles and seed board together.')
    report = {'finite':finite()}
    if args.tiles:
        report['original'] = original(args.tiles,args.seed_board,args.checkpoint,args.seconds)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2)
            stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

