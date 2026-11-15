"""Separate endpoint and physical-strand cuts lazily and verify spherical embeddings."""
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import time

from r27_structure_filters import Union, domain_bits
from r28_exchange import read_json
from r28_geo_color import signed_phase_model
from r28_geo_sewing import side_options
from r29_gold_geography import checked_domains
from r29_gold_phase import EndpointCuts, checked_pins, _reciprocal
from search_problem import SearchProblem


def _groups(links):
    adjacency = defaultdict(set)
    for a,b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen=set()
    groups=[]
    for a in adjacency:
        if a in seen:
            continue
        pending=[a]
        seen.add(a)
        group=set()
        while pending:
            v=pending.pop()
            group.add(v)
            for w in adjacency[v]-seen:
                seen.add(w)
                pending.append(w)
        groups.append(frozenset(group))
    return sorted(groups,key=lambda g:(len(g),min(g)))


def _occurs(m, choices):
    if len(choices)==0:
        return False
    # An occurrence is either unconditional or a disjunction of phase literals.
    if any(type(x) is bool and x for x in choices):
        return True
    out=m.new_bool_var("occupied")
    m.add_bool_or(choices).only_enforce_if(out)
    for x in choices:
        m.add_implication(x,out)
    return out


# These small partitions were separated on the released puzzle. They are
# structural hints only: re-evaluate the crossing bound from this puzzle's
# physical strands each time, for both choices of global color complement.
PUBLISHED_ENDPOINT_SETS=(
    ((66,1,2),(66,1,7)),
    ((40,1,4),(40,1,6),(260,1,3)),
)


def insufficient_crossings(data, pins, groups=PUBLISHED_ENDPOINT_SETS):
    """A necessary cut property, recomputed for every permitted tile phase.

    The maxima may choose a different phase on every tile. This relaxation
    is sound for any root or branch, including phase pins derived by parity.
    """
    for row in groups:
        group=frozenset(tuple(v) for v in row)
        # Flipping global colors may exchange canonical endpoint keys with
        # their reciprocals. Rebuild the complementary keys from the masks.
        def flip(key):
            m,c,k=key
            c^=1
            return min((m,c,k),(*_reciprocal(m,c),12-k))
        for subset in (group,frozenset(flip(v) for v in group)):
            if not subset or not subset.issubset(data.graph):
                continue
            inside=outside=False
            upper=0
            for tile,phases in enumerate(data.strands):
                opts=(phases[pins[tile]],) if tile in pins else phases
                crossings=[]
                inside_here=outside_here=True
                for strands in opts:
                    used={v for pair in strands for v in pair}
                    inside_here &= bool(used & subset)
                    outside_here &= bool(used - subset)
                    crossings.append(sum((a in subset)!=(b in subset) for a,b in strands))
                inside |= inside_here
                outside |= outside_here
                upper+=max(crossings)
                if upper>=2:
                    break
            if upper<2 and inside and outside:
                return dict(part=sorted(subset),upper=upper,required=2)
    return None


def endpoint_constraint(m, phases, data, group):
    """A closed tour meeting both sides crosses the type partition at least twice."""
    inside,outside=[],[]
    crossings=[]
    for t,pair in enumerate(data.strands):
        occupied = [[],[]]
        counts=[]
        for row in pair:
            endpoints={v for ab in row for v in ab}
            occupied[0].append(bool(endpoints & group))
            occupied[1].append(bool(endpoints - group))
            counts.append(sum((a in group)!=(b in group) for a,b in row))
        for opts,sides in zip(occupied,(inside,outside)):
            if all(opts):
                sides.append(True)
            elif any(opts):
                sides.append(phases[t] if opts[1] else ~phases[t])
        crossings.append(counts[0]+(counts[1]-counts[0])*phases[t])
    used=[_occurs(m,inside),_occurs(m,outside)]
    if any(type(x) is bool and not x for x in used):
        return
    m.add(sum(crossings)>=2).only_enforce_if([x for x in used if type(x) is not bool])


def _all_seam_edges(problem, data, domains):
    from validator import reverse_mask
    if domains==domain_bits(problem):
        # Actual color and opposite traversal are imposed separately below.
        groups=defaultdict(list)
        for a,(mask,_) in enumerate(data.sides):
            groups[mask].append(a)
        return [(a,b) for a,(mask,_) in enumerate(data.sides)
                for b in groups[reverse_mask(mask)] if a<b and a//3!=b//3]
    options=side_options(problem,domains)
    return [(a,b) for a,row in enumerate(options) for b in sorted(row) if a<b]


def embedding(problem, domains, mates):
    """All physical sides paired. Every pose is determined by one root pose."""
    if len(mates)!=3*problem.n or any(v<0 or v>=len(mates) or mates[v]!=a for a,v in enumerate(mates)):
        raise ValueError("Expected a complete reciprocal mate vector.")
    for cell in range(problem.n):
        for r in range(3):
            if not (domains[cell]>>r)&1:
                continue
            board={cell:r}
            placements={0:(cell,r)}
            pending=[0]
            failed=False
            while pending and not failed:
                tile=pending.pop()
                c,rot=placements[tile]
                for s in range(3):
                    other,os=divmod(mates[3*tile+s],3)
                    b,j=problem.board.neighbors[c][(s+rot)%3]
                    pose=(b,(j-os)%3)
                    code=3*other+pose[1]
                    if other in placements:
                        if placements[other]!=pose:
                            failed=True
                            break
                    elif b in board or not ((domains[b]>>code)&1):
                        failed=True
                        break
                    else:
                        pending.append(other)
                        placements[other]=pose
                        board[b]=code
            if not failed and len(board)==problem.n:
                return [board[i] for i in range(problem.n)]
    return None


class GoldMatching:
    def __init__(self,problem,domains,pins,bonds,cutsets=(),forbidden=(),*,physical_support=True,project_blanks=False,corner_mode=None,neighbor_mode=None):
        self.problem=problem
        self.domains=domains
        self.inputs=EndpointCuts(problem)
        self.m,self.phase=signed_phase_model(problem,pins=pins,bonds=bonds)
        self.type_seen=set()
        self.strand_seen=set()
        offsets=[0]
        for tile in problem.tiles:
            offsets.append(offsets[-1]+len(tile['segments']))
        self.strands=offsets[-1]
        self.mate={}
        self.transitions={}
        options=_all_seam_edges(problem,self.inputs,domains)
        banned={tuple(sorted(x)) for x in forbidden}
        wanted={tuple(sorted(x)) for x in bonds}
        if physical_support:
            from r32_gold_bonds import inventory
            physical = inventory(problem)
            banned.update(physical.forbidden)
            wanted.update(physical.required)
            if physical.closed_surface and (bonds or forbidden or domains != domain_bits(problem)):
                from r27_structure_filters import Work, WorkLimit
                from r32_gold_bonds import gold_matching_supports
                fixed = dict(wanted)
                fixed.update({b: a for a, b in wanted})
                rows = [set() for _ in range(3*problem.n)]
                for a, b in options:
                    if (a,b) not in banned and fixed.get(a,b) == b and fixed.get(b,a) == a:
                        rows[a].add(b)
                        rows[b].add(a)
                try:
                    supported = gold_matching_supports(problem, rows, Work(200000))
                except WorkLimit:
                    supported = rows
                options = [] if supported is None else [
                    (a,b) for a,b in options if b in supported[a]]
        self.neighbor_stats = None
        if neighbor_mode:
            from r35_physical_adjacency import filter_neighbors
            allowed,self.neighbor_stats = filter_neighbors(problem,set(options)-banned,wanted,
                                                           connect=neighbor_mode=='connectivity')
            options=sorted(allowed)
        self.corner_state = None
        self.corner_stats = None
        if corner_mode:
            from r35_corner_chains import filter_options
            allowed,self.corner_state,self.corner_stats = filter_options(problem,set(options)-banned,wanted,
                                                                        packing=corner_mode!='chains')
            options=sorted(allowed)
        self.blank_projection = None
        if project_blanks and any(not mask for mask,_ in self.inputs.sides):
            from r35_blank_projection import BlankProjection
            self.blank_projection = BlankProjection(self.inputs.sides,
                                                   (pair for pair in options if pair not in banned), wanted)
            if not self.blank_projection.possible:
                self.m.add_bool_or([])
        by_side=defaultdict(list)
        for a,b in options:
            if (a,b) in banned or (self.blank_projection is not None and not self.inputs.sides[a][0]):
                continue
            bit=self.m.new_bool_var("physical_seam")
            self.mate[a,b]=bit
            by_side[a].append(bit)
            by_side[b].append(bit)
            mask, color=self.inputs.sides[a]
            flip=color ^ self.inputs.sides[b][1] ^ (mask.bit_count()&1)
            self.m.add(self.phase[a//3] != self.phase[b//3] if flip else
                       self.phase[a//3] == self.phase[b//3]).only_enforce_if(bit)
            links=[]
            for k in range(1,12):
                if (mask>>(k-1))&1:
                    t,i=self.inputs.endpoints[a,k]
                    u,j=self.inputs.endpoints[b,12-k]
                    links.append((offsets[t]+i, offsets[u]+j))
            self.transitions[a,b]=links
        for a in range(3*problem.n):
            if self.blank_projection is None or self.inputs.sides[a][0]:
                self.m.add_exactly_one(by_side[a])
        for pair in wanted:
            if self.blank_projection is not None and pair in self.blank_projection.required:
                continue
            if pair in self.mate:
                self.m.add(self.mate[pair]==1)
            else:
                self.m.add_bool_or([])
        if neighbor_mode:
            from r35_physical_adjacency import post_tile_limits
            self.neighbor_stats['pair_bounds']=post_tile_limits(self)
        if corner_mode=='positions':
            from r35_corner_chains import post_positions
            self.corner_stats['positions'] = post_positions(self)
        if self.blank_projection is not None:
            self.required_nonblank = wanted.intersection(self.mate)
        for group in cutsets:
            self.strand_cut(group)

    def strand_cut(self, group):
        group=frozenset(group)
        if (any(type(i) is not int or not 0<=i<self.strands for i in group)
                or not 0<len(group)<self.strands):
            raise ValueError("A cut needs a nonempty proper set of physical strands.")
        complement=frozenset(range(self.strands))-group
        if min(complement)<min(group):
            group=complement
        if group in self.strand_seen:
            return
        self.strand_seen.add(group)
        terms=[]
        for pair, links in self.transitions.items():
            count=sum((u in group)!=(v in group) for u,v in links)
            if count:
                terms.append(count*self.mate[pair])
        self.m.add(sum(terms)>=2)

    def crossing_capacity(self, group, seconds):
        """Maximum cut crossings in one simultaneous phase-compatible perfect matching."""
        import math
        from ortools.sat.python import cp_model
        group=frozenset(group)
        if self.blank_projection is not None:
            raise ValueError("Exact physical crossing capacity requires the full side model.")
        if seconds<=0 or any(type(i) is not int or not 0<=i<self.strands for i in group) or not 0<len(group)<self.strands:
            raise ValueError("Invalid physical cut or capacity budget.")
        m=self.m.clone()
        objective=sum(sum((u in group)!=(v in group) for u,v in links)*self.mate[pair]
                      for pair,links in self.transitions.items())
        m.maximize(objective)
        solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=1
        solver.parameters.max_time_in_seconds=seconds
        status=solver.solve(m)
        out=dict(status=solver.status_name(status),branches=solver.num_branches)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            out.update(value=round(solver.objective_value),upper=math.ceil(solver.best_objective_bound))
        return out

    def process(self, solver, geography, lazy_types, stats, charge_cut, deadline=float("inf")):
        values=[solver.value(x) for x in self.phase]
        type_groups=_groups(ab for t,v in enumerate(values) for ab in self.inputs.strands[t][v])
        if lazy_types and len(type_groups)>1:
            changed=False
            for group in type_groups:
                if group not in self.type_seen:
                    charge_cut(group)
                    endpoint_constraint(self.m,self.phase,self.inputs,group)
                    self.type_seen.add(group)
                    changed=True
            if not changed:
                raise RuntimeError("Endpoint partition failed to exclude its disconnected witness.")
            stats['type_cuts']=len(self.type_seen)
            return None
        active=[p for p,x in self.mate.items() if solver.value(x)]
        parent=Union(self.strands)
        for p in active:
            for u,v in self.transitions[p]:
                parent.join(u,v)
        groups=defaultdict(set)
        for i in range(self.strands):
            groups[parent.find(i)].add(i)
        if len(groups)>1:
            for g in groups.values():
                self.strand_cut(g)
            stats['strand_cuts']=len(self.strand_seen)
            return None
        if self.corner_state is not None:
            conflict = self.corner_state.conflict(active)
            if conflict is not None:
                self.m.add_bool_or([~self.mate[p] for p in conflict])
                stats['corner_sewing_cuts'] += 1
                return None
        if self.blank_projection is not None:
            from r35_blank_projection import lift
            from r31_geographic import physical_bonds
            blank = self.blank_projection
            signed = blank.signed(values)
            phase_block = [~self.phase[t] if values[t] else self.phase[t] for t in blank.owners]
            stats['nonblank_witnesses_checked'] += 1
            # If every nonblank join was fixed or is the only remaining option,
            # this circuit and its relative tile phases are the entire scoped search.
            sole_sewing = len(active) == len(self.mate) or self.required_nonblank.issuperset(active)
            if not geography:
                completion = blank.matching(signed,deadline,stats)
                if completion is not None:
                    return dict(status="CONNECTED_SEWING",pairs=sorted([*active,*completion]),phases=values)
                # Blank compatibility depends only on its owners' phases and fixed inputs.
                stats['blank_phase_failures'] += 1
                if sole_sewing:
                    return dict(status="INFEASIBLE",physical_cutsets=[sorted(g) for g in self.strand_seen])
                self.m.add_bool_or(phase_block)
                return None
            codes = lift(self.problem,self.domains,active,blank,signed,deadline,stats)
            if codes is not None:
                if not self.problem.validate(codes)['valid']:
                    raise ArithmeticError("Projected sewing failed independent Gold verification.")
                pairs = physical_bonds(self.problem,codes)
                return dict(status="GOLD",codes=codes,phases=values,pairs=pairs)
            if sole_sewing:
                stats['failed_embeddings'] += 1
                return dict(status="INFEASIBLE",physical_cutsets=[sorted(g) for g in self.strand_seen])
            if getattr(self, "geo_conflicts", None) is not None:
                from r35_blank_projection import check_time
                check_time(deadline)
                core = self.geo_conflicts.learn(active)
                if core is not None:
                    self.m.add_bool_or([~self.mate[p] for p in core])
                    stats["geographic_core_sizes"].append(len(core))
                    stats["geographic_cores"] = [sorted(c) for c in self.geo_conflicts.cores]
                    stats['failed_embeddings'] += 1
                    return None
            # normalize_tiles requires a strand on every tile. Its connected
            # nonblank graph therefore fixes all phases up to global complement.
            self.m.add_bool_or([~self.mate[p] for p in active])
            stats['failed_embeddings'] += 1
            return None
        if not geography:
            return dict(status="CONNECTED_SEWING",pairs=active,phases=values)
        mates=[-1]*(3*self.problem.n)
        for a,b in active:
            mates[a],mates[b]=b,a
        codes=embedding(self.problem,self.domains,mates)
        if codes is not None:
            if not self.problem.validate(codes)['valid']:
                raise ArithmeticError("Decoded physical matching failed independent Gold verification.")
            return dict(status="GOLD",codes=codes,phases=values,pairs=active)
        core = self.geo_conflicts.learn(active) if getattr(self, "geo_conflicts", None) is not None else None
        block = active if core is None else core
        self.m.add_bool_or([~self.mate[p] for p in block])
        if core is not None:
            stats["geographic_core_sizes"].append(len(core))
            stats["geographic_cores"] = [sorted(c) for c in self.geo_conflicts.cores]
        stats['failed_embeddings']+=1
        return None


def search(problem, *, domains=None, pins=None, bonds=(), forbidden=(), cutsets=(), matching=True,
           geography=False, max_rounds=32, seconds=5, lazy_types=True, type_cuts=(),
           learned_geometry=False, channel=False, fraction=False, bridges=(), geographic_cores=(), departure_rules=(), physical_support=True, physical_loss=None, project_blanks=False, corner_mode=None, neighbor_mode=None):
    from ortools.sat.python import cp_model
    if seconds<=0 or max_rounds<1 or (geography and not matching):
        raise ValueError("Invalid Gold solve budget or mode.")
    pins=None if pins is None else checked_pins(problem.n,pins)
    domains=checked_domains(problem,domains)
    for a,b in [*bonds,*forbidden]:
        if type(a) is not int or type(b) is not int or not 0<=a<3*problem.n or not 0<=b<3*problem.n:
            raise ValueError("Invalid physical side index.")
    started=time.monotonic()
    stats=dict(cp_calls=0,branches=0,type_cuts=0,strand_cuts=0,failed_embeddings=0)
    if neighbor_mode not in (None,'local','connectivity') or (neighbor_mode and not geography):
        raise ValueError("Physical tile-neighbor constraints require an embed search.")
    if corner_mode not in (None,'chains','packing','positions') or (corner_mode and not geography):
        raise ValueError("Corner constraints require an embed search and a supported corner mode.")
    if type(project_blanks) is not bool:
        raise ValueError("Expected a boolean blank-side projection option.")
    if project_blanks and not matching:
        raise ValueError("Blank-side projection requires a physical matching search.")
    if matching:
        # These optional full-side formulas refer directly to individual blank literals.
        # Preserve their constraints whenever one of them is requested.
        if project_blanks and (channel or fraction or bridges or physical_loss is not None or corner_mode=='positions'):
            stats['blank_projection_fallback'] = 'full-side channel, pricing, bridge, reference-loss, or vertex position model'
            project_blanks = False
        check=GoldMatching(problem,domains,pins,bonds,cutsets,forbidden,physical_support=physical_support,
                           project_blanks=project_blanks,corner_mode=corner_mode,neighbor_mode=neighbor_mode)
        if neighbor_mode:
            stats['tile_neighbor_preprocessing']=check.neighbor_stats
        if corner_mode:
            stats.update(corner_preprocessing=check.corner_stats,corner_sewing_cuts=0)
        if check.blank_projection is not None:
            stats.update(blank_pairs_deferred=len(check.blank_projection.allowed), nonblank_witnesses_checked=0,
                         blank_phase_failures=0, blank_matching_states=0, blank_root_trials=0)
        if physical_loss is not None:
            from r32_gold_loss import add_loss_budget
            reference, spec, cores = physical_loss
            add_loss_budget(check, reference, spec)
            from r32_gold_loss import Certificates
            replay = Certificates(problem)
            replay.add(cores)
            for core in replay.rows:
                if core <= check.mate.keys():
                    check.m.add_bool_or([~check.mate[p] for p in core])
        if (learned_geometry or geographic_cores or channel) and not geography:
            raise ValueError("Geographic exclusions require the embed search objective.")
        if learned_geometry or geographic_cores:
            from r31_geographic import GeographicConflicts
            check.geo_conflicts=GeographicConflicts(problem,domains)
            stats.update(geographic_core_sizes=[],geographic_cores=[])
            for core in geographic_cores:
                pairs=frozenset(tuple(sorted(row)) for row in core)
                if pairs<=check.mate.keys() and check.geo_conflicts.impossible(pairs):
                    m=check.m
                    m.add_bool_or([~check.mate[p] for p in pairs])
                    check.geo_conflicts.cores.append(pairs)
            stats['geographic_cores']=[sorted(c) for c in check.geo_conflicts.cores]
        if channel:
            from r31_geographic import channel_geographic
            stats['channel']=channel_geographic(check)

        if fraction:
            from r31_gold_pricing import fractional_cuts
            stats['fractional']=fractional_cuts(check,bonds,pins)
        if departure_rules:
            from r31_geographic import add_departure_rules
            stats["rechecked_departure_cores"]=add_departure_rules(check,departure_rules)
        if bridges:
            from r31_geographic import tighten_bridge
            distinct=sorted({tuple(sorted(set(group))) for group in bridges},key=lambda g:(len(g),g))
            stats['bridges']=[tighten_bridge(check,g,min(seconds,2)) for g in distinct]
            stats['bridge_hints']=[g for g,r in zip(distinct,stats['bridges']) if r['status']=='CHECKED']
        m=check.m
        stats['strand_cuts']=len(check.strand_seen)
        stats['partner_variables']=len(check.mate)
    else:
        if cutsets or forbidden:
            raise ValueError("Physical strand groups and forbidden seams require matching.")
        m,phase=signed_phase_model(problem,pins=pins,bonds=bonds)
        inputs=EndpointCuts(problem)
        seen=set()
    data=check.inputs if matching else inputs
    phase_vars=check.phase if matching else phase
    seen_types=check.type_seen if matching else seen
    fresh=[]
    reusable=[]
    def charge_cut(group):
        ordered=sorted(group)
        fresh.append(ordered)
        reusable.append(ordered)
    for vertices in type_cuts:
        group=frozenset(map(tuple,vertices))
        if not group or group-set(data.graph):
            raise ValueError("Invalid endpoint type cut.")
        if group not in seen_types:
            endpoint_constraint(m,phase_vars,data,group)
            seen_types.add(group)
            reusable.append(sorted(group))
    stats['reused_type_cuts']=len(seen_types)
    stats['discovered_type_sets']=fresh
    stats['reusable_type_sets']=reusable
    for _ in range(max_rounds):
        remaining=seconds-(time.monotonic()-started)
        if remaining<=0:
            break
        s=cp_model.CpSolver()
        s.parameters.num_search_workers=1
        s.parameters.max_time_in_seconds=remaining
        status=s.solve(m)
        stats['cp_calls']+=1
        stats['branches']+=s.num_branches
        if status==cp_model.MODEL_INVALID:
            raise RuntimeError(s.solution_info())
        if status==cp_model.INFEASIBLE:
            if matching:
                stats["physical_cutsets"]=[sorted(g) for g in sorted(check.strand_seen,key=lambda s:(len(s),tuple(sorted(s))))]
            return dict(status="INFEASIBLE",**stats)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            break
        if matching:
            from r35_blank_projection import ProjectionTimeout
            try:
                result=check.process(s,geography,lazy_types,stats,charge_cut,started+seconds)
            except ProjectionTimeout:
                break
            if result:
                return dict(result,**stats)
        else:
            values=[s.value(x) for x in phase]
            groups=_groups(ab for t,v in enumerate(values) for ab in inputs.strands[t][v])
            if len(groups)==1:
                return dict(status="CONNECTED_TYPES",phases=values,**stats)
            for g in groups:
                if g in seen:
                    continue
                charge_cut(g)
                endpoint_constraint(m,phase,inputs,g)
                seen.add(g)
            stats['type_cuts']=len(seen)
    return dict(status="UNKNOWN",**stats)


def restriction_core(problem, *, domains=None, pins=None, bonds=(), forbidden=(), seconds=4, max_delete=16):
    """Recheck simultaneous-connectivity failure while releasing proposed restrictions."""
    pins={} if pins is None else pins
    rows=([('pin',t,v) for t,v in pins.items()]+[('bond',a,b) for a,b in bonds]
          +[('forbid',a,b) for a,b in forbidden])
    def test(kept):
        return search(problem,domains=domains,pins={a:b for k,a,b in kept if k=='pin'},
                      bonds=[(a,b) for k,a,b in kept if k=='bond'],
                      forbidden=[(a,b) for k,a,b in kept if k=='forbid'],
                      seconds=seconds,matching=True,geography=False)
    if max_delete<0:
        raise ValueError("Invalid core deletion budget.")
    first=test(rows)
    if first['status']!='INFEASIBLE':
        return dict(status=first['status'],queries=1)
    calls=1
    for row in rows.copy()[:max_delete]:
        reduced=rows.copy()
        reduced.remove(row)
        calls+=1
        if test(reduced)['status']=='INFEASIBLE':
            rows=reduced
    confirm=test(rows)
    return dict(status=confirm['status'],queries=calls+1,binding=restriction_binding(problem,domains),
                restrictions=rows if confirm['status']=='INFEASIBLE' else None)


def restriction_binding(problem,domains):
    rows=checked_domains(problem,domains)
    return hashlib.sha256(json.dumps([problem.fingerprint,list(map(hex,rows))]).encode()).hexdigest()


def core_applies(problem, domains, pins, bonds, forbidden, rule):
    if rule.get('status')!='INFEASIBLE' or rule.get('binding')!=restriction_binding(problem,domains):
        return False
    actual={('pin',i,v) for i,v in (pins or {}).items()}
    actual.update(('bond',*sorted(pair)) for pair in bonds)
    actual.update(('forbid',*sorted(pair)) for pair in forbidden)
    for k,a,b in rule['restrictions']:
        if (k,a,b) not in actual and (k=='pin' or (k,b,a) not in actual):
            return False
    return True


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("method",choices=("types","matching","embed","core","capacity"))
    cli.add_argument("--tiles",default="data/tiles.json")
    cli.add_argument("--input",type=Path)
    cli.add_argument("--seconds",type=float,default=5)
    cli.add_argument("--rounds",type=int,default=32)
    cli.add_argument("--neighbor-mode",choices=("local","connectivity"),
                     help="Check tile-neighbor multiplicity/Hall support and optionally target-mesh connectivity.")
    cli.add_argument("--corner-mode",choices=("chains","packing","positions"),
                     help="Check partial physical vertex rings when embedding a sewing.")
    cli.add_argument("--project-blanks",action="store_true",
                     help="Defer blank-side choices in matching or embed; infer geographic blank joins from the mesh.")
    cli.add_argument("--output",type=Path,help="Save a result and its reusable cuts.")
    cli.add_argument("--reuse",type=Path,help="Reuse a checked physical core or endpoint cut sets.")
    cli.add_argument("--departure-branch",type=int,help="Check one disjoint first-broken-bond branch from a departure result.")
    args=cli.parse_args()
    data=read_json(args.input) if args.input else {}
    project_blanks=args.project_blanks or data.get('project_blanks',False)
    corner_mode=args.corner_mode or data.get('corner_mode')
    neighbor_mode=args.neighbor_mode or data.get('neighbor_mode')
    if neighbor_mode and args.method!='embed':
        cli.error('--neighbor-mode requires embed.')
    if corner_mode and args.method!='embed':
        cli.error('--corner-mode requires embed.')
    if project_blanks and args.method not in ('embed','matching'):
        cli.error('--project-blanks requires matching or embed.')
    problem=SearchProblem(read_json(args.tiles))
    opts=dict(domains=None if 'domains' not in data else
              tuple(sum(1<<v for v in set(row)) for row in data['domains']),
              pins=None if 'pins' not in data else {int(t):v for t,v in data['pins'].items()},
              bonds=data.get('bonds',()),forbidden=data.get('forbidden',()),seconds=args.seconds)
    prior=read_json(args.reuse) if args.reuse else {}
    if args.departure_branch is not None:
        from r29_gold_phase import SignedParity
        cores=[sorted({tuple(sorted(p)) for p in row}) for row in prior.get("cores",())]
        if args.method not in ("matching","embed") or not cores:
            cli.error("--departure-branch requires matching or embed and --reuse of a departure result.")
        core=min(cores,key=lambda row:(len(row),row))
        parity=SignedParity(problem)
        if all(parity.bond(a,b) for a,b in core):
            cli.error("The first physical departure core fails rechecking on this puzzle.")
        index=args.departure_branch
        if not 0<=index<len(core):
            cli.error("Departure branch index is out of range.")
        opts['bonds']=[*opts['bonds'],*core[:index]]
        opts['forbidden']=[*opts['forbidden'],core[index]]
    if args.method=='capacity':
        checker=GoldMatching(problem,checked_domains(problem,opts['domains']),opts['pins'],
                             opts['bonds'],(),opts['forbidden'])
        out=[]
        for group in data.get('strand_cuts',()):
            # Successive capacities include every previously required crossing.
            out.append(checker.crossing_capacity(group,args.seconds))
            checker.strand_cut(group)
    elif args.method=='core':
        out=restriction_core(problem,**opts)
    elif args.method!='types' and 'restrictions' in prior and core_applies(problem,**{k:v for k,v in opts.items() if k!='seconds'},rule=prior):
        out=dict(status="INFEASIBLE",reason="certified-physical-core",cp_calls=0)
    else:
        reusable=prior.get('reusable_type_sets',prior.get('discovered_type_sets',()))
        out=search(problem,matching=args.method!='types',geography=args.method=='embed',
                   cutsets=data.get('strand_cuts',()),type_cuts=reusable,max_rounds=args.rounds,
                   learned_geometry=args.method=="embed" and data.get("learn_geography",True),
                   channel=data.get("channel",args.method=="embed" and not project_blanks and not corner_mode and not neighbor_mode),
                   corner_mode=corner_mode,neighbor_mode=neighbor_mode,
                   project_blanks=project_blanks,fraction=data.get("fractional",False),
                   bridges=[*data.get("bridges",()),*prior.get("bridge_hints",())],
                   geographic_cores=prior.get("geographic_cores",()) if args.method=="embed" else (),
                   departure_rules=prior.get("cores",()),**opts)
    if args.departure_branch is not None:
        out["covered_departure_branch"]={"required":core[:index],"forbidden":core[index],"total":len(core)}
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,out)
    print(json.dumps(out))


if __name__=="__main__":
    main()
