"""Condition a reciprocal side-class perfect matching on one or more selected bonds."""
from collections import defaultdict
from functools import lru_cache

from r27_structure_filters import _components


def conditional_matches(darts,options,*,max_states=30000):
    """Return necessary and possible co-selections; a capped calculation makes no claim."""
    nodes=tuple(sorted(darts));index={d:i for i,d in enumerate(nodes)}
    edges=tuple(sorted(options));incident=[[] for u in nodes]
    for i,(a,b) in enumerate(edges):
        u,v=index[a],index[b]
        incident[u].append((v,i));incident[v].append((u,i))
    full=(1<<len(nodes))-1
    states=0
    @lru_cache(None)
    def solve(remaining):
        nonlocal states
        states+=1
        if states>max_states:raise OverflowError
        if not remaining:return 0,0
        vertices=[u for u in range(len(nodes)) if remaining>>u&1]
        u=min(vertices,key=lambda v:sum(remaining>>w&1 for w,_ in incident[v]))
        mandatory=None;possible=0
        for v,i in incident[u]:
            if not remaining>>v&1:continue
            child=solve(remaining^(1<<u)^(1<<v))
            if child is None:continue
            m,p=child;bit=1<<i
            mandatory=m|bit if mandatory is None else mandatory&(m|bit)
            possible|=p|bit
        return None if mandatory is None else (mandatory,possible)
    def decode(row):
        while row:
            bit=row&-row; row^=bit
            yield edges[bit.bit_length()-1]
    try:
        root=solve(full)
        if root is None:return dict(complete=True,feasible=False,states=states)
        global_must,global_may=root
        bundles={};incompatible={}
        for i,(a,b) in enumerate(edges):
            if not global_may>>i&1 or global_must>>i&1:continue
            fixed=solve(full^(1<<index[a])^(1<<index[b]))
            forced,possible=fixed
            others=tuple(decode(forced & ~global_must))
            if others:bundles[a,b]=others
            excluded=tuple(e for e in decode(global_may&~possible) if a not in e and b not in e)
            if excluded:incompatible[a,b]=excluded
        return dict(complete=True,feasible=True,states=states,bundles=bundles,incompatible=incompatible)
    except OverflowError:
        return dict(complete=False,states=states)


def correlated_sides(size,options):
    graph=[set() for _ in range(size)]
    for a,b in options:graph[a].add(b);graph[b].add(a)
    together={};exclusions={};sizes=[]
    for group in _components(graph):
        if not group:continue
        if len(group)>18 and min(len(graph[v]) for v in group)>3:
            sizes.append((len(group),False,0));continue
        pairs={(u,v) for u in group for v in graph[u] if u<v}
        calc=conditional_matches(group,pairs)
        sizes.append((len(group),calc['complete'],calc['states']))
        if calc.get('feasible'):
            together.update(calc['bundles'])
            exclusions.update(calc['incompatible'])
    return together,exclusions,sizes


def price(problem,options,required,pins=None,*,geography=False):
    """Reject an optional mate only when its whole conditional bundle fails."""
    from collections import Counter
    from r29_gold_phase import SignedParity
    from r32_gold_bonds import endpoints,loop_groups
    from r35_corner_chains import CornerChains,degree_inventory
    from r35_turn_words import TurnWords

    options=set(options);required=set(required)
    owners=Counter(d for e in options for d in e)
    required.update(e for e in options if any(owners[d]==1 for d in e))
    bundles,exclusions,sizes=correlated_sides(3*problem.n,options)
    stats=dict(bundle_heads=len(bundles),positive_implications=sum(map(len,bundles.values())),
               nonincident_exclusions=sum(map(len,exclusions.values())),
               classes_priced=sum(ok for _,ok,_ in sizes),class_states=sum(s for _,_,s in sizes))
    rejected=set();why=Counter();examples=[]
    if not bundles:return rejected,stats
    arcs=endpoints(problem)
    sides=[[] for _ in range(3*problem.n)]
    for a,k in arcs:sides[a].append(k)
    parity=SignedParity(problem)
    for t,z in (pins or {}).items():parity.pin(t,z)
    for e in sorted(required):parity.bond(*e)
    mark=len(parity.base.trail)
    degrees=degree_inventory(problem) if geography else None
    corners=CornerChains(problem.n,degrees) if degrees else None
    words=TurnWords(problem) if degrees else None
    def conflict(extra):
        edges=required|set(extra)
        if len({d for pair in edges for d in pair})!=2*len(edges):return 'ownership'
        parity.base.rollback(mark)
        if parity.base.conflict is not None or any(not parity.bond(*e) for e in sorted(extra)):return 'phase'
        if loop_groups(arcs,sides,edges,problem.total_segments):return 'proper_loop'
        if corners and corners.conflict(sorted(edges)):return 'corner'
        if words and words.conflicts(edges,maximum=1):return 'word'
        return None
    baseline=conflict(())
    if baseline:
        stats['already_failed']=baseline
        return rejected,stats
    tested={}
    for e,implied in bundles.items():
        if conflict((e,)):continue
        together=frozenset((e,*implied))
        if together not in tested:tested[together]=conflict(together)
        reason=tested[together]
        if reason:
            rejected.add(e);why[reason]+=1
            if len(examples)<6:examples.append(dict(head=e,implied=implied,conflict=reason))
    stats.update(unique_joint_checks=len(tested),extra_rejections=dict(why),examples=examples)
    return rejected,stats


def post_correlations(check,pins=(),required=(),*,signed_only=False):
    """Post exact conditional consequences within signed, compatible physical classes."""
    if signed_only and not pins or any(n<0 for row in check.problem.neighbors for n in row):return {}
    points=check.inputs.sides
    pins=pins or {}
    fixed=dict(required)
    fixed.update((b,a) for a,b in required)
    options=set()
    represented=set(check.mate)
    available=represented|set(check.blank_projection.allowed) if check.blank_projection is not None else represented
    for a,b in available:
        if fixed.get(a,b)!=b or fixed.get(b,a)!=a:continue
        mask,ca=points[a];_,cb=points[b]
        if a//3 in pins and b//3 in pins and pins[a//3]^pins[b//3]!=ca^cb^(mask.bit_count()&1):continue
        options.add((a,b))
    if pins:
        # Conditional signed classes can contain forced partners before the
        # native signed integer model has chosen a location for either owner.
        from r27_structure_filters import Work,WorkLimit
        from r32_gold_bonds import gold_matching_supports
        graph=[set() for _ in range(3*check.problem.n)]
        for a,b in options:graph[a].add(b);graph[b].add(a)
        try:rows=gold_matching_supports(check.problem,graph,Work(200000))
        except WorkLimit:rows=graph
        if rows is None:
            check.m.add_bool_or([])
            return dict(signed_matching_infeasible=True)
        options={(a,b) for a,row in enumerate(rows) for b in row if a<b}
    implications,forbidden,sizes=correlated_sides(3*check.problem.n,options)
    excluded=set()
    if pins and implications:
        from collections import Counter
        from r29_gold_phase import SignedParity
        degree=Counter(d for e in options for d in e)
        basis=SignedParity(check.problem)
        for t,z in pins.items():basis.pin(t,z)
        for edge in sorted(set(required)|{e for e in options if any(degree[d]==1 for d in e)}):basis.bond(*edge)
        mark=len(basis.base.trail)
        for head,others in implications.items():
            basis.base.rollback(mark)
            if head in represented and basis.bond(*head) and any(not basis.bond(*e) for e in others):
                excluded.add(head)
                check.m.add(check.mate[head]==0)
    if signed_only:return dict(signed_bundle_exclusions=sorted(excluded),table_states=sum(s for _,_,s in sizes))
    negatives={tuple(sorted((e,f))) for e,others in forbidden.items() for f in others if e in represented and f in represented}
    directed={(e,f) for e,others in implications.items() for f in others if e in represented and f in represented}
    equalities={(e,f) for e,f in directed if e<f and (f,e) in directed}
    for a,b in equalities:check.m.add(check.mate[a]==check.mate[b])
    conditional={(e,f) for e,f in directed if (f,e) not in directed}
    for a,b in conditional:check.m.add_implication(check.mate[a],check.mate[b])
    for a,b in negatives:check.m.add_bool_or([~check.mate[a],~check.mate[b]])
    return dict(signed_bundle_exclusions=sorted(excluded),equalities=len(equalities),directional=len(conditional),nonincident_exclusions=len(negatives),
                table_states=sum(s for _,_,s in sizes))
