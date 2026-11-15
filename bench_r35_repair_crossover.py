"""Check which new physical claims strengthen the archived geographic repair master."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations, product
from pathlib import Path
from random import Random
from unittest.mock import patch

from bench_r35_repair_seeds import independently_allowed, PoseOracle
from constraint_lns import score
from constraint_pruning import Pruner
from io_utils import atomic_json
from r28_exchange import read_json
from r35_geographic_packing_check import check_packing
from r35_phase_sewing import strand_components
from r35_repair_seeds import active_catalog, repair_rows
from search_problem import SearchProblem
from verify_r35_path_mates import VectorReplay

DOCS=Path(__file__).resolve().parent/'docs'


class PlacementProjection:
    """One unchanged seed cell fixes its entire compulsory physical component in any Gold board."""
    def __init__(self, problem, codes, required, cover):
        self.problem=problem
        self.cover=sorted(map(frozenset,cover),key=lambda row:(len(row),sorted(row)))
        self.targets=defaultdict(list)
        links=defaultdict(list)
        for a,b in required:
            links[a//3].append((a%3,b//3,b%3))
            links[b//3].append((b%3,a//3,a%3))
        for anchor,code in enumerate(codes):
            known={code//3:(anchor,code%3)}
            pending=[code//3]
            while pending:
                tile=pending.pop()
                cell,turn=known[tile]
                for side,other,back in links[tile]:
                    dest,facing=problem.board.neighbors[cell][(side+turn)%3]
                    pose=(dest,(facing-back)%3)
                    if dest<0 or other in known and known[other]!=pose:
                        raise ValueError('Unexpected inconsistent required component in published mesh.')
                    if other not in known:
                        known[other]=pose
                        pending.append(other)
            if len({c for c,_ in known.values()})!=len(known):
                raise ValueError('Compulsory component overlaps itself.')
            for tile,(cell,turn) in known.items():
                self.targets[tile].append((cell,turn,anchor))

    def options(self,edge):
        a,b=edge
        choices=set()
        for cell,rotation,anchor in self.targets[a//3]:
            dest,facing=self.problem.board.neighbors[cell][(a%3+rotation)%3]
            for other,turn,second in self.targets[b//3]:
                if (other,(b%3+turn)%3)==(dest,facing):
                    choices.add(frozenset((anchor,second)))
        return choices

    def compare(self,options):
        """Enumerate all minimal anchor unions; domination is monotone under additional premises."""
        if any(not group for group in options):
            return dict(applicable=False,missing=sum(not group for group in options))
        choices={frozenset()}
        dominated=set()
        for group in sorted(options,key=len):
            new=sorted({earlier|part for earlier in choices for part in group},key=lambda r:(len(r),sorted(r)))
            choices=set()
            for row in new:
                if any(old<=row for old in choices):
                    continue
                proof=next((old for old in self.cover if old<=row),None)
                if proof is not None:
                    dominated.add(tuple(sorted(proof)))
                else:
                    choices.add(row)
            if not choices:
                break
        return dict(applicable=True,existing_subsets=[list(r) for r in sorted(dominated)],
                    undominated=[sorted(row) for row in sorted(choices,key=lambda r:(len(r),sorted(r)))])


def checked_subsets(problem, candidates, old_rows, codes, allowed, witnesses):
    """Test every subset below the undominated projected premises with independent unit propagation."""
    oracle=PoseOracle(problem,allowed)
    subsets={frozenset(part) for group in candidates for k in range(1,len(group)+1)
             for part in combinations(group,k)}
    proved=set()
    try:
        for subset in sorted(subsets,key=lambda q:(len(q),sorted(q))):
            if not any(old<=subset for old in proved) and oracle.replay(codes,subset):
                proved.add(subset)
    finally:
        oracle.close()
    result=[]
    fresh=sorted((group for group in proved if not any(row<=group for row in old_rows)),
                 key=lambda q:(len(q),sorted(q)))
    if {tuple(sorted(group)) for group in fresh}!=set(witnesses):
        raise ValueError('Independent minimal placement conflicts differ from the witnessed ones.')
    for group in fresh:
        bits=int(witnesses[tuple(sorted(group))],16)
        hit={c for c in range(problem.n) if bits>>c&1}
        if bits>>problem.n or group&hit or not all(row&hit for row in old_rows):
            raise ValueError('Claimed previous-master selection does not support the comparison.')
        result.append(dict(cells=sorted(group),changed=len(hit),previous_master_hit=hex(bits)))
    return dict(subsets=len(subsets),proved_minimal=len(proved),baseline_rows=len(old_rows),
                undominated=result)


def verify_overlaps(problem,codes,allowed,envelopes,baseline,claims,record,exhaust):
    """Recheck every accepted row and every older-master selection said to be eliminated."""
    from hashlib import sha256
    if sha256((DOCS/'R35_REPAIR_REGIONAL.json').read_bytes().strip()).hexdigest()!=record['source_catalog_sha256']:
        raise ValueError('Region catalogue differs from the independent comparison.')
    small={tuple(g) for g in claims['small']}
    larger={tuple(g) for g in claims['larger']}
    old={tuple(sorted(g)) for g in baseline}
    def covered(q,rows):
        return any(t in rows for k in range(1,len(q)+1) for t in combinations(q,k))
    def hit(mask):
        number=int(mask,16)
        if number<0 or number>>problem.n:raise ValueError('Invalid geographic Hamming selection.')
        return {c for c in range(problem.n) if number>>c&1}
    primary=[hit(q) for q in record['small_hits']]
    secondary=[hit(q) for q in record['large_hits']]
    if (not all(all(g.intersection(h) for g in baseline) for h in primary+secondary)
            or not all(all(set(g)&h for g in small) for h in secondary)
            or not all(any(not set(g)&h for h in primary) for g in small)
            or not all(any(not set(g)&h for h in secondary) for g in larger)):
        raise ValueError('An accepted conflict fails to remove a previously surviving actual selection.')
    oracle=PoseOracle(problem,allowed)
    try:
        for row in small|larger:
            if (covered(row,old) or not oracle.replay(codes,row)
                    or any(oracle.replay(codes,row[:i]+row[i+1:]) for i in range(len(row)))):
                raise ValueError('A published minimal anchor conflict is redundant or unsupported.')
        if exhaust:
            saved=old.copy()
            found=set()
            metrics=[]
            for k in range(1,max(map(len,envelopes[:19]))+1):
                candidates=sorted({s for i,g in enumerate(envelopes) if i!=19 or k<=4
                                   for s in combinations(sorted(g),k)})
                measure=Counter(total=len(candidates))
                for row in candidates:
                    if covered(row,saved):measure['dominated']+=1
                    else:
                        measure['sat_checks']+=1
                        if oracle.replay(codes,row):
                            found.add(row);saved.add(row);measure['new']+=1
                metrics.append(dict(width=k,**measure))
            if found!=small or metrics!=record['metrics']:
                raise ValueError('Complete small region/root census disagrees.')
        comparisons=record['larger_scope']
        donor=set(envelopes[19])
        evaluated={}
        rejected=0
        for bits in comparisons['selections_bits']:
            h=hit(bits)
            if not all(g&h for g in baseline) or not all(set(g)&h for g in small):
                raise ValueError('Held-out larger-scope selection fails the prior master.')
            held=tuple(sorted(donor-h))
            if held not in evaluated:evaluated[held]=oracle.replay(codes,held)
            failed=evaluated[held]
            if failed!=any(not set(g)&h for g in larger):
                raise ValueError('Stored large rows and actual held-out geographic oracle disagree.')
            rejected+=failed
        if (len(comparisons['selections_bits'])!=len(set(comparisons['selections_bits'])) or
                rejected!=comparisons['selections']-comparisons['surviving']):
            raise ValueError('Incomplete held-out comparisons.')
        return record
    finally:
        oracle.close()


def check(problem, donor, witnesses, *, overlap=None, exhaust=False):
    # Establish both the 3,924-row archive and the public full master before this extension.
    with patch('r35_repair_seeds.regional_catalog',return_value=None):
        catalog=active_catalog(problem)
        old_runtime=repair_rows(problem,catalog['codes']) if catalog else ()
    if catalog is None:
        raise ValueError('The published board and repair evidence must match.')
    codes=catalog['codes']
    if score(problem,codes)['matched_edges']!=207:
        raise ValueError('Unexpected archived score.')
    allowed,physical=independently_allowed(problem)
    required=frozenset(map(tuple,physical['verified_required']))
    projection=PlacementProjection(problem,codes,required,catalog['rows'])
    vector=VectorReplay(problem)
    envelopes=[]
    def project(edges):
        options=[projection.options(edge) for edge in sorted(edges-required)]
        if all(options):
            envelopes.append(frozenset(c for alternatives in options for support in alternatives for c in support))
        return projection.compare(options)
    result={'puzzle':problem.fingerprint,'cover_rows':len(catalog['rows']),'path':[]}
    path=read_json(DOCS/'R35_PATH_CIRCUITS.json')
    if path['fingerprint']!=problem.fingerprint or donor['fingerprint']!=problem.fingerprint:
        raise ValueError('Different source mesh.')
    for candidate in path['gold_path_circuits']:
        named=frozenset(map(tuple,candidate['path']))
        mate=tuple(candidate['mate'])
        loop=frozenset(map(tuple,candidate['loop']))
        _,matched,misses=vector.count(named,[{mate}])
        if (mate not in loop or not matched or any(misses) or
                not any(0<v<problem.total_segments for v in strand_components(problem,loop,complete=False))):
            raise ValueError('Invalid no-good from path-induced circuit.')
        result['path'].append(project(named|(loop-{mate})))
    result['donor']=[]
    for raw in donor['geographic_cores']:
        named=frozenset(map(tuple,raw))
        replay=check_packing(problem,named)
        if replay['feasible'] is not False:
            raise ValueError('Geographic exclusion did not replay.')
        result['donor'].append(dict(edges=len(named),required=len(required&named),
                                    independent_nodes=replay['nodes'],**project(named)))
    contacts=read_json(DOCS/'R35_POSE_CONTACTS.json')
    if contacts['original']['input']!=problem.fingerprint:
        raise ValueError('Different physical mesh.')
    result['forbidden']=[]
    for rows in contacts['original']['conditional_banned'].values():
        for source in rows:
            edge=tuple(source['bond'])
            if edge in allowed:
                raise ValueError('Held-out forbidden mate was independently admitted.')
            result['forbidden'].append(dict(edge=edge,**project({edge})))
    result['nine_tiles']=[]
    for source in contacts['all_nine_trees']:
        if source['gold_incidental'].get('0'):
            continue
        members=set(source['members'])
        named=frozenset(map(tuple,source['edges']))
        complete=named|{edge for edge in required if all(d//3 in members for d in edge)}
        _,matched,_=vector.count(complete,[])
        if matched:
            raise ValueError('An alleged impossible nine-tile component has a raw compatible root.')
        result['nine_tiles'].append(dict(seed=source['seed'],**project(complete)))
    clusters=Pruner(problem,regions=[],parity=False,gold_bonds=True).clusters
    result['four_dominoes']=[]
    for source in contacts['original']['cases']:
        if source['kind']!='four_local_dominoes' or source['runs']['joint']['possible']:
            continue
        rng=Random(source['seed'])
        patch_cells={rng.randrange(problem.n)}
        for _ in range(3):
            patch_cells.update(b for c in list(patch_cells) for b,_ in problem.board.neighbors[c] if b>=0)
        families=[]
        forced=[]
        for _,_,embeddings in clusters[:4]:
            chosen=rng.sample([p for p in embeddings if all(c in patch_cells for c,_ in p)],4)
            families.append(chosen)
            options=set()
            for pose in chosen:
                # Both members are one compulsory domino. Either endpoint determines the pose.
                for cell,code in pose:
                    options.update(frozenset((anchor,)) for loc,turn,anchor in projection.targets[code//3]
                                   if (loc,turn)==(cell,code%3))
            forced.append(options)
        for entry in product(*families):
            flat=[pair for pose in entry for pair in pose]
            place=dict(flat)
            if len(place)!=len(flat):
                continue
            for c,s,b,t in problem.edges:
                if c not in place or b not in place:
                    continue
                first,second=place[c],place[b]
                e=tuple(sorted((3*(first//3)+(s-first%3)%3,3*(second//3)+(t-second%3)%3)))
                if e not in allowed:
                    break
            else:
                raise ValueError('Four-table unrestricted pose combination survived.')
        result['four_dominoes'].append(dict(seed=source['seed'],**projection.compare(forced)))
    candidates=[row for item in result['path'] for row in item.get('undominated',())]
    baseline=set(old_runtime)|{frozenset(row) for row in catalog['rows']}
    result['envelopes']=[sorted(g) for g in envelopes]
    result['subsets']=checked_subsets(problem,candidates,baseline,codes,allowed,witnesses)
    if overlap is not None:
        if len(envelopes)!=22 or len(envelopes[19])!=29:
            raise ValueError('A different regional corpus requires fresh projection.')
        claims=read_json(DOCS/'R35_REPAIR_REGIONAL.json')
        result['overlap']=verify_overlaps(problem,codes,allowed,envelopes,baseline,claims,overlap,exhaust)
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--donor',type=Path,default=DOCS/'R35_MIP_DONOR_SEWINGS.json')
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--witnesses',type=Path,default=DOCS/'R35_REPAIR_CROSSOVER.json')
    cli.add_argument('--exhaust-overlap',action='store_true',help='Enumerate every subset in each path region and donor subsets through four cells.')
    cli.add_argument('--write-cnf',type=Path,help='Regenerate the verified maximum-excluded change formula.')
    args=cli.parse_args()
    saved=read_json(args.witnesses)
    previous=saved['subsets']['undominated']
    selections={tuple(row['cells']):row['previous_master_hit'] for row in previous}
    problem=SearchProblem(read_json(args.tiles))
    result=check(problem,read_json(args.donor),selections,overlap=saved.get('overlap'),exhaust=args.exhaust_overlap)
    complete=active_catalog(problem)
    hint=set(complete['hit'])
    if not all(hint.intersection(row) for row in complete['rows']):
        raise ValueError('The published master upper assignment violates a placement group.')
    if args.write_cnf:
        from bench_r35_repair_refinement import write_formula
        write_formula(problem.n,complete['rows'],complete['lower_bound']-1,args.write_cnf)
    result['lower_bound']=complete['lower_bound']
    result['upper']=len(hint)
    atomic_json(args.output,result)
