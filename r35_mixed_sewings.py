"""Search complete alternating physical seam choices with a phase for each offspring."""
from collections import Counter,deque

from r27_structure_filters import domain_bits
from r30_gold import embedding


def geographic_roots(problem,domains,edges):
    """Exclude a root only when allowed physical mates empty a geographic cell domain."""
    if len(domains)!=problem.n:
        raise ValueError('Expected one pose domain per board cell.')
    mates=[set() for _ in range(3*problem.n)]
    for a,b in edges:
        mates[a].add(b)
        mates[b].add(a)
    supports={}
    for s in range(3):
        for j in range(3):
            row=[]
            for code in range(3*problem.n):
                tile,rot=divmod(code,3)
                d=3*tile+(s-rot)%3
                row.append(sum(1<<(3*(m//3)+(j-m%3)%3) for m in mates[d]))
            supports[s,j]=row
    roots=visits=0
    survivors=[]
    for home,d in enumerate(domains):
        for rot in range(3):
            if not d&(1<<rot):continue
            roots+=1
            work=deque([home])
            current=list(domains)
            current[home]=1<<rot
            viable=all(current)
            while viable and work:
                cell=work.popleft()
                choices=current[cell]
                # Large domains offer weak supports. A skipped revision can only
                # leave extra roots to be checked by the exact search.
                if choices.bit_count()>32:continue
                visits+=1
                for side,(other,opposite) in enumerate(problem.board.neighbors[cell]):
                    if other<0:continue
                    masks=supports[side,opposite]
                    cells=choices
                    allowed=0
                    while cells:
                        code=cells&-cells
                        allowed|=masks[code.bit_length()-1]
                        cells-=code
                    nxt=current[other]&allowed
                    if not nxt:
                        viable=False
                        break
                    if nxt!=current[other]:
                        current[other]=nxt
                        work.append(other)
            if viable:survivors.append((home,rot))
    return dict(checked_roots=roots,survivors=survivors,cell_revisions=visits)


def donor_geography(problem,records,domains=None):
    """Test a relaxation containing every matching assembled from any named donor."""
    from r29_gold_phase import EndpointCuts
    from r35_alternating_sewings import complete
    if not records:raise ValueError('At least one complete donor is required.')
    sides=EndpointCuts(problem)
    allowed=set()
    for record in records:
        mate=complete(problem,record['pairs'],record['phases'],sides)
        allowed.update((a,b) for a,b in mate.items() if a<b)
    result=geographic_roots(problem,domain_bits(problem) if domains is None else domains,allowed)
    return dict(parent_count=len(records),physical_seams=len(allowed),roots=result,
                geographic_status='UNKNOWN' if result['survivors'] else 'INFEASIBLE')


class PhaseComponents:
    """Rollback tile components with the XOR from each tile to its representative."""
    def __init__(self,n):
        from r29_logic_binary import Parity
        self.parity=Parity(n)
        self.size=[1]*n
        self.log=[]

    def push(self,equations):
        p=self.parity
        for a,b,bit in equations:
            u,x=p.find(a)
            v,y=p.find(b)
            diff=bit^x^y
            if u==v:
                if diff:return False
                continue
            if self.size[u]<self.size[v]:u,v=v,u
            self.log.append((v,u))
            p.p[v]=u
            p.sign[v]=diff
            self.size[u]+=self.size[v]
        return True

    def restore(self,mark):
        while len(self.log)>mark:
            v,u=self.log.pop()
            self.size[u]-=self.size[v]
            self.parity.p[v]=v
            self.parity.sign[v]=0

    def frontier(self,tiles):
        labels={}
        result=[]
        for tile in tiles:
            root,bit=self.parity.find(tile)
            label,base=labels.setdefault(root,(len(labels),bit))
            result.append(2*label+(base^bit))
        return tuple(result)


def mixed_solve(family,*,max_nodes=1000000,memo=True,geography=False,domains=None,fixed=None,exclude_phases=(),lookahead=48,repeat_forcing=False):
    """Count each signed, connected full child once regardless of its number of phase choices."""
    from r35_alternating_sewings import RollbackPaths
    p=family.problem
    domains=domain_bits(p) if domains is None else domains
    fixed={} if fixed is None else dict(fixed)
    all_groups=family.groups+family.blanks
    if (type(max_nodes) is not int or max_nodes<0 or
        type(lookahead) is not int or lookahead<0 or
        any(type(i) is not int or not 0<=i<len(all_groups) or type(v) is not int or v not in (0,1)
            for i,v in fixed.items())):
        raise ValueError('Invalid mixed parent search limits or cycle restrictions.')
    base=set(family.fixed)
    for i,which in fixed.items():base.update(all_groups[i][which])
    choices=sorted((i for i in range(len(all_groups)) if i not in fixed),
                   key=lambda i:(-len(all_groups[i][0]),i))
    possible=1<<len(choices)
    geo=None
    if geography:
        allowed=base.union(*(set(half) for i in choices for half in all_groups[i]))
        geo=geographic_roots(p,domains,allowed)
        if not geo['survivors']:
            return dict(complete=True,geographic_status='INFEASIBLE',geographic_roots=geo,
                        possible_full_sewings=possible,connected_full=None,gold=None,
                        work={'nodes':0},examples=[])
    def equations(pairs):
        data=family.inputs.sides
        return tuple((a//3,b//3,data[a][1]^data[b][1]^(data[a][0].bit_count()&1)) for a,b in pairs)
    ph=PhaseComponents(p.n)
    paths=RollbackPaths(family.n)
    link_options=[*family.options,*[((),()) for _ in family.blanks]]
    phase_options=[tuple(equations(half) for half in group) for group in all_groups]
    fronts={}
    tiles=set()
    strands=set()
    for i in reversed(choices):
        tiles.update(v for a,b,_ in phase_options[i][0] for v in (a,b))
        strands.update(v for a,b in link_options[i][0] for v in (a,b))
        fronts[i]=tuple(sorted(tiles)),tuple(sorted(strands))
    def violates(phase,pairs):
        return any((phase[a]^phase[b])!=bit for a,b,bit in pairs)
    if any(len(phase)!=p.n or any(bit not in (0,1) for bit in phase) for phase in exclude_phases):
        raise ValueError('Invalid excluded physical phase.')
    base_eq=equations(base)
    def conflicts(rows):
        return sum(1<<i for i,phase in enumerate(exclude_phases) if violates(phase,rows))
    startflags=conflicts(base_eq)
    flags_by_half=[tuple(conflicts(row) for row in pair) for pair in phase_options] if exclude_phases else ()
    need=(1<<len(exclude_phases))-1
    tail=[0]*(len(choices)+1)
    if need:
        for j in reversed(range(len(choices))):
            one,two=flags_by_half[choices[j]]
            tail[j]=tail[j+1]|one|two
    common=PhaseComponents(p.n)
    common_phase=None
    if common.push(base_eq) and all(common.push(half) for i in choices for half in phase_options[i]):
        common_phase=[common.parity.find(t)[1] for t in range(p.n)]
    work=Counter()
    store={}
    words=0
    examples=[]
    gold=None
    stopped=False
    startmask=sum(v<<i for i,v in fixed.items())
    masklo=(1<<len(family.groups))-1
    def accept(i,which):
        if common_phase is None and not ph.push(phase_options[i][which]):
            return 'phase_conflicts'
        if not paths.push(link_options[i][which]):
            return 'proper_loops'
        return None

    def visit(j,mask,flags,assigned=0):
        nonlocal words,stopped,gold
        while assigned&(1<<j):
            assigned^=1<<j
            j+=1
        if (flags|tail[j])!=need:
            work['covered_phase_scopes']+=1
            return 0
        if work['nodes']>=max_nodes:
            stopped=True
            return 0
        work['nodes']+=1
        if j==len(choices):
            if len(examples)<4:
                examples.append(dict(mask=mask,phases=common_phase if common_phase is not None else
                                     [ph.parity.find(t)[1] for t in range(p.n)]))
            if geography:
                mesh=family.pairs(mask&masklo,mask>>len(family.groups))
                mate=[-1]*(3*p.n)
                for a,b in mesh:mate[a]=b;mate[b]=a
                gold=embedding(p,domains,mate)
                if gold is not None and not p.validate(gold)['valid']:
                    raise ValueError('Mixed offspring geographic witness failed Gold validation.')
            return 1
        i=choices[j]
        key=None
        if memo and not geography:
            tiles,strands=fronts[i]
            key=(j,assigned,flags,() if common_phase is not None else ph.frontier(tiles),paths.frontier(strands))
            if key in store:
                work['cache_hits']+=1
                return store[key]
        total=0
        initial_phase,initial_path=len(ph.log),len(paths.log)
        # Test later whole cycles against the current path and phase. A doomed
        # half stays doomed after any compatible extension.
        viable=True
        while True:
            changed=False
            for pos in range(j+1,min(len(choices),j+1+lookahead)):
                if assigned&(1<<pos):continue
                probe=choices[pos]
                good=[]
                for half in (0,1):
                    phase_mark,path_mark=len(ph.log),len(paths.log)
                    work['lookahead_half_tests']+=1
                    if accept(probe,half) is None:good.append(half)
                    ph.restore(phase_mark)
                    paths.restore(path_mark)
                if not good:
                    work['dead_lookaheads']+=1
                    viable=False
                    break
                if len(good)==1:
                    half=good[0]
                    assigned|=1<<pos
                    mask|=half<<probe
                    if need:flags|=flags_by_half[probe][half]
                    if accept(probe,half) is not None:
                        raise ArithmeticError('A forced choice changed within a single probe.')
                    work['forced_cycles']+=1
                    changed=True
            if not (changed and viable and repeat_forcing):break
        if viable:
            for which in (0,1):
                phase_mark,path_mark=len(ph.log),len(paths.log)
                work['decision_half_tests']+=1
                reason=accept(i,which)
                if reason:
                    work[reason]+=1
                else:
                    further_flags=flags|flags_by_half[i][which] if need else 0
                    total+=visit(j+1,mask|(which<<i),further_flags,assigned)
                paths.restore(path_mark)
                ph.restore(phase_mark)
                if stopped or gold is not None:break
        paths.restore(initial_path)
        ph.restore(initial_phase)
        if key is not None and not stopped and words+len(key[3])+len(key[4])+1<=2000000:
            store[key]=total
            words+=1+len(key[3])+len(key[4])
        return total
    fixed_links=[*family.base,*(edge for i,which in fixed.items() for edge in link_options[i][which])]
    valid=family.n and ph.push(base_eq) and paths.push(fixed_links)
    total=visit(0,startmask,startflags) if valid else 0
    return dict(complete=not stopped and gold is None,possible_full_sewings=possible,
                connected_full=total,gold=gold,examples=examples,geographic_roots=geo,
                geographic_status=('GOLD' if gold is not None else 'UNKNOWN' if stopped else 'INFEASIBLE') if geography else None,
                work=dict(work),memo_entries=len(store))
