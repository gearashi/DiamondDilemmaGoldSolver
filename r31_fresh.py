"""Independent starts and bounded repair experiments, with independently validated boards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from constraint_pruning import Pruner
from constraint_symmetry import proper_rotations
from r27_incumbent import BoardModel, assignment
from r28_exchange import domain_binding, read_json
from r28_exchange_patches import AssignmentPrices
from r30_fresh import fresh_start
from search_problem import SearchProblem


def physical_bonds(model, board):
    mask = model.matches(board)
    left = board[model.a[mask]]
    right = board[model.b[mask]]
    a = 3*(left//3) + (model.sa[mask]-left%3)%3
    b = 3*(right//3) + (model.sb[mask]-right%3)%3
    return frozenset(zip(np.minimum(a,b).tolist(), np.maximum(a,b).tolist()))


def quality(result):
    return (result['gold'], result['score'], -result['components'])


class Assemblies:
    """Pose entire bonded components. Remaining identities complete a legal bijection."""
    def __init__(self, model, *, gold_bonds=True):
        self.model = model
        if gold_bonds:
            from r32_gold_bonds import inventory
            self.forbidden = inventory(model.problem).forbidden
        else:
            self.forbidden = frozenset()
        rules = Pruner(model.problem, regions=[], parity=False, gold_bonds=gold_bonds)
        self.bonds = frozenset(tuple(sorted((3*a+s,3*b+t))) for a,s,b,t in rules.bonds)
        self.groups = []
        self.tiles = set()
        for tiles, _, embeddings in rules.clusters:
            fits = [tuple((int(c),int(v)) for c,v in row) for row in embeddings
                    if all(model.legal[c,v] for c,v in row)]
            self.groups.append((set(tiles), fits))
            self.tiles.update(tiles)
        self.groups.sort(key=lambda item:len(item[1]))

    def protected(self, board):
        return np.array([j for j,v in enumerate(board) if v//3 in self.tiles], dtype=int)

    def seed(self, hint, rng, work=20000):
        model = self.model
        hint = model.legal_board(hint)
        bonds = physical_bonds(model,hint)
        if self.bonds <= bonds and not self.forbidden & bonds:
            return hint
        poses = []
        for _, fits in self.groups:
            # Preserve already present physical poses first. Otherwise try a fresh embedding.
            order = rng.permutation(len(fits))
            poses.append(sorted((fits[i] for i in order),
                                key=lambda row:-sum(hint[c]==v for c,v in row)))
        pending = [0]
        def place(level, fixed):
            if pending[0]>=work:
                return None
            if level < len(poses):
                for row in poses[level]:
                    if any(c in fixed for c,_ in row):
                        continue
                    pending[0]+=1
                    child=place(level+1, fixed | dict(row))
                    if child is not None or pending[0]>=work:
                        return child
                return None
            board=hint.copy()
            for c,v in fixed.items():
                board[c]=v
            cells=np.array(sorted(set(range(model.n))-fixed.keys()),dtype=int)
            tiles=np.array(sorted(set(range(model.n))-self.tiles),dtype=int)
            if len(cells):
                codes=3*tiles[:,None]+np.arange(3)
                allowed=model.legal[cells[:,None,None],codes[None,:,:]]
                weights=rng.integers(0,8,size=allowed.shape)
                # One surviving sampled placement outweighs every random tie on this board.
                weights+=(8*model.n+1)*(hint[cells,None,None]==codes[None,:,:])
                weights[~allowed]=-1
                try:
                    donors=assignment(weights.max(axis=2),allowed.any(axis=2))
                except ValueError:
                    return None
                rotations=weights[np.arange(len(cells)),donors].argmax(axis=1)
                board[cells]=codes[donors,rotations]
            return self.clear_forbidden(model.legal_board(board), rng)
        board=place(0,{})
        if board is None:
            raise ValueError("No legal mandatory-bond start found within the pose budget.")
        if not self.bonds <= physical_bonds(model,board):
            raise ArithmeticError("Mandatory-bond construction failed verification.")
        return board

    def clear_forbidden(self, board, rng):
        """Repair forbidden physical mates while preserving all obligatory assemblies."""
        if not self.forbidden:
            return board
        model = self.model
        banned = self.forbidden & physical_bonds(model, board)
        if not banned:
            return board
        allowed = [int(c) for c in model.free if board[c]//3 not in self.tiles]
        for _ in range(len(banned)):
            a,b = min(banned)
            affected = [c for c in allowed if board[c]//3 in (a//3,b//3)]
            found = False
            for c in affected:
                for d in [c, *rng.permutation(allowed)]:
                    tc,td = board[c]//3,board[d]//3
                    for r in range(3):
                        candidate = board.copy()
                        candidate[c] = 3*td+r
                        if c != d:
                            candidate[d] = 3*tc+board[c]%3
                        if not model.legal[c,candidate[c]] or not model.legal[d,candidate[d]]:
                            continue
                        remaining = self.forbidden & physical_bonds(model,candidate)
                        if remaining < banned:
                            board,banned,found = candidate,remaining,True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None
            if not banned:
                break
        return board

    def relocate(self, board, rng, trials=8):
        if not self.groups:
            return board
        model=self.model
        group,poses=self.groups[int(rng.integers(len(self.groups)))]
        old=set(int(c) for c in np.flatnonzero(np.isin(board//3,list(group))))
        blocked=set(self.protected(board))-old
        best=board
        best_score=int(model.matches(board).sum())
        for index in rng.permutation(len(poses))[:trials]:
            pose=dict(poses[index])
            if pose.keys() & blocked:
                continue
            empty=sorted(old-pose.keys())
            displaced=sorted(pose.keys()-old)
            candidate=board.copy()
            if empty:
                tiles=board[displaced]//3
                codes=3*tiles[:,None]+np.arange(3)
                legal=model.legal[np.array(empty)[:,None,None],codes[None,:,:]]
                prices=np.where(legal,rng.integers(64,size=legal.shape),-1)
                try:
                    donors=assignment(prices.max(axis=2),legal.any(axis=2))
                except ValueError:
                    continue
                rot=prices[np.arange(len(empty)),donors].argmax(axis=1)
                candidate[empty]=codes[donors,rot]
            for c,v in pose.items():
                candidate[c]=v
            score=int(model.matches(candidate).sum())
            if score>=best_score and not self.forbidden & physical_bonds(model,candidate):
                best,best_score=candidate,score
        return best


def joint_repair(model, values, cells, *, adjacent=False, effort=0.05, contacts=None,
                 disjoint=False):
    """Choose donor cycles, their support and rotations in one bounded CP model."""
    from ortools.sat.python import cp_model
    limit = (-1 if disjoint else int(adjacent)) if contacts is None else contacts
    if type(limit) is not int or limit < -1:
        raise ValueError("Contact limit must be nonnegative or -1 for unrestricted.")
    allow_contacts = limit != 0
    board=model.legal_board(values)
    cells=tuple(sorted(set(map(int,cells))))
    if any(c<0 or c>=model.n or c in model.problem.fixed for c in cells):
        raise ValueError('Expected a pool of movable cells.')
    if not cells:
        return board,{'status':'EMPTY','branches':0}
    cp=cp_model.CpModel()
    local={c:i for i,c in enumerate(cells)}
    inventory=board[list(cells)]//3
    codes=3*inventory[:,None]+np.arange(3)
    flat=codes.ravel()
    donors=[]
    labels={}
    changed={}
    terms=[]
    for i,c in enumerate(cells):
        options=np.flatnonzero(model.legal[c,flat]).tolist()
        old=3*i+int(board[c]%3)
        label=cp.new_int_var_from_domain(cp_model.Domain.from_values(options),'')
        donor=cp.new_int_var(0,len(cells)-1,'')
        cp.add_division_equality(donor,label,3)
        donors.append(donor)
        flag=cp.new_bool_var('')
        cp.add(label!=old).only_enforce_if(flag)
        cp.add(label==old).only_enforce_if(~flag)
        cp.add_hint(label,old)
        cp.add_hint(donor,i)
        cp.add_hint(flag,False)
        labels[c],changed[c]=label,flag
        gain=np.zeros(len(flat),dtype=int)
        for s,neighbor in enumerate(model.problem.neighbors[c]):
            if neighbor<0 or (allow_contacts and neighbor in local):
                continue
            side=model.problem.sides[c,s]
            gain+=(model.masks[flat,s]==model.reverse[model.masks[board[neighbor],side]])
        # Nonadjacent changes have additive incident seam score deltas.
        gain-=gain[old]
        if gain.max()!=gain.min():
            value=cp.new_int_var(int(gain.min()),int(gain.max()),'')
            cp.add_element(label,gain.tolist(),value)
            terms.append(value)
    cp.add_all_different(donors)
    shared=[]
    incident={c:[] for c in cells}
    for a,s,b,t in model.problem.edges:
        if a not in local or b not in local:
            continue
        if not allow_contacts:
            cp.add_at_most_one(changed[a],changed[b])
            continue
        if limit > 0 or disjoint:
            both=cp.new_bool_var('')
            cp.add_bool_and([changed[a],changed[b]]).only_enforce_if(both)
            cp.add_bool_or([~changed[a],~changed[b],both])
            shared.append(both)
            if disjoint:
                incident[a].append(both)
                incident[b].append(both)
        left=cp.new_int_var(0,2047,'')
        right=cp.new_int_var(0,2047,'')
        cp.add_element(labels[a],model.masks[flat,s].tolist(),left)
        cp.add_element(labels[b],model.reverse[model.masks[flat,t]].tolist(),right)
        match=cp.new_bool_var('')
        cp.add(left==right).only_enforce_if(match)
        cp.add(left!=right).only_enforce_if(~match)
        terms.append(match)
    if shared and limit >= 0:
        cp.add(sum(shared)<=limit)
    if disjoint:
        for edges in incident.values():
            if len(edges)>1:
                cp.add_at_most_one(edges)
    cp.maximize(sum(terms))
    solver=cp_model.CpSolver()
    solver.parameters.max_deterministic_time=effort
    solver.parameters.num_search_workers=1
    status=solver.solve(cp)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        candidate=board.copy()
        for c in cells:
            candidate[c]=flat[solver.value(labels[c])]
        candidate=model.legal_board(candidate)
        if model.matches(candidate).sum()>=model.matches(board).sum():
            board=candidate
    return board,dict(status=solver.status_name(status),branches=int(solver.num_branches))


def result(model, board):
    values=model.legal_board(board).tolist()
    check=model.problem.validate(values)
    return dict(codes=values, gold=bool(check['valid']),score=int(check['matched_edges']),
                components=int(check['line_components']))


def tie_repair(model, board, cells, *, slack=0, cap=8):
    prices=AssignmentPrices(model)
    prices.update(board)
    choices={}
    for gap in range(slack+1):
        allowance=cap//(slack+1)+int(gap<cap%(slack+1))
        if not allowance:
            continue
        for row in prices.independent(cells,slack=gap,min_slack=gap,limit=allowance):
            r=result(model,row)
            if r['gold']:
                return row,1+sum(len(rows) for rows in choices.values())
            choices.setdefault(r['score'],[]).append(r)
    if not choices:
        return board,0
    top=max(choices)
    best=max(choices[top],key=quality)
    # Admit a lower score only if it strictly improves the endpoint connectivity.
    for score in sorted(choices,reverse=True):
        row=max(choices[score],key=quality)
        if row['components']<best['components']:
            best=row
    return np.array(best['codes']),sum(len(rows) for rows in choices.values())


def relink(model, first, second, assemblies, limit=6):
    """Sample shortest-path internal transpositions in both directions, after alignment."""
    a=model.legal_board(first)
    b=model.legal_board(second)
    try:
        actions=proper_rotations(model.problem.board)
    except ValueError:
        actions=()
    candidates=[b]
    for rot in actions:
        row=np.array(rot.apply(b.tolist()))
        if np.all(model.legal[np.arange(model.n),row]):
            candidates.append(row)
    b=max(candidates,key=lambda row:int(np.sum(a//3==row//3)))
    seen={a.tobytes(),b.tobytes()}
    out=[]
    for start, target in ((a,b),(b,a)):
        row=start.copy()
        chain=[]
        for c in range(model.n):
            if row[c]//3==target[c]//3:
                continue
            donor=int(np.flatnonzero(row//3==target[c]//3)[0])
            row[c],row[donor]=int(target[c]),int(row[c])
            if not model.legal[donor,row[donor]]:
                options=[3*(row[donor]//3)+r for r in range(3) if model.legal[donor,3*(row[donor]//3)+r]]
                if not options:
                    break
                row[donor]=options[0]
            key=row.tobytes()
            if key in seen:
                continue
            if assemblies:
                actual = physical_bonds(model,row)
                if not assemblies.bonds <= actual or assemblies.forbidden & actual:
                    continue
            seen.add(key)
            chain.append(row.copy())
        if chain:
            for j in np.unique(np.linspace(0,len(chain)-1,min(limit,len(chain)),dtype=int)):
                out.append(chain[j])
    return out


def simultaneous_swaps(model, board, allowed, prices, *, exact=False, touch=None, packing_work=None, separator_depth=0):
    """Compare all two-cycles and pack compatible positive gains across the full board."""
    from r35_swap_catalog import SwapCatalog
    cells=np.array(sorted(allowed),dtype=int)
    if not len(cells):
        return board,0
    prices.update(board)
    catalog=getattr(prices,'swap_catalog',None)
    if catalog is None or catalog.model is not model:
        catalog=prices.swap_catalog=SwapCatalog(model)
    ranked=catalog.rank(prices,board,cells)
    left,right,new_left,new_right,weights=ranked
    count=len(weights)
    active=range(count)
    if exact:
        from r34_swap_packing import pack_swaps
        active=catalog.noncontacting(ranked,separator_depth,packing_work,pack_swaps)
    excluded=set()
    selected=[]
    candidate=board.copy()
    for slot in active:
        a,b=left[slot],right[slot]
        if a in excluded or b in excluded:
            continue
        candidate[a],candidate[b]=new_left[slot],new_right[slot]
        excluded.update((a,b,*model.problem.neighbors[a],*model.problem.neighbors[b]))
        if touch:
            selected.append(slot)
    if touch:
        from r35_touch_packing import augment
        pairs=list(zip(left.tolist(),right.tolist()))
        gain=weights.tolist()
        oriented=list(zip(new_left.tolist(),new_right.tolist()))
        candidate,_,metrics=augment(model,board,pairs,oriented,gain,selected,touch)
        if packing_work is not None:
            packing_work.update({'touch_'+k:v for k,v in metrics.items()})
    return candidate,count


def pack_cycles(model, board, cells, prices, *, relaxations=3):
    """Bound the whole nonadjacent repair, splitting an edge whose two endpoints changed.

    A permutation relaxation can link disconnected geographic cells in long donor cycles.
    Its optimal score bounds every independent support with these pins. Each integral
    cycle with an independent support can also supply a feasible primal improvement.
    """
    import heapq
    cells=np.array(sorted(cells),dtype=int)
    if not len(cells):
        return board,dict(relaxations=0,complete=True,cycles=0)
    prices.update(board)
    grid=prices.prices[cells][:,board[cells]//3]
    rotation=grid.argmax(axis=2)
    gain=grid.max(axis=2).astype(int)
    old=grid[np.arange(len(cells)),np.arange(len(cells)),board[cells]%3]
    gain-=old[:,None]
    # Favor the unchanged orientation on a tight identity arc.
    for i in range(len(cells)):
        if gain[i,i]==0:
            rotation[i,i]=board[cells[i]]%3
    legal=grid.max(axis=2)>=0
    queue=[(-3*len(cells),())]
    visited={()}
    best=board.copy()
    best_gain=solves=cycles_seen=0
    while queue and solves<relaxations:
        parent,pinned=heapq.heappop(queue)
        if -parent<=best_gain:
            continue
        allowed=legal.copy()
        weights=gain.copy()
        for i in pinned:
            allowed[i]=False
            allowed[i,i]=True
            weights[i,i]=0
        donors=assignment(weights,allowed)
        solves+=1
        upper=int(weights[np.arange(len(cells)),donors].sum())
        if upper<=best_gain:
            continue
        # Convert permutation to cycles, including profitable in-place reorientations.
        unseen=set(range(len(cells)))-set(pinned)
        cycles=[]
        while unseen:
            i=min(unseen)
            cycle=[]
            while i in unseen:
                cycle.append(i)
                unseen.remove(i)
                i=int(donors[i])
            value=sum(int(weights[i,donors[i]]) for i in cycle)
            if value>0:
                cycles.append((value,cycle))
        blocked=set()
        primal=board.copy()
        primal_gain=0
        for weight,cycle in sorted(cycles,key=lambda item:-item[0]):
            footprint=set(cells[cycle])
            if footprint&blocked:
                continue
            neighbors=set(model.problem.neighbors[cells[cycle]].ravel())
            if footprint&neighbors:
                continue
            for i in cycle:
                code=3*(board[cells[donors[i]]]//3)+rotation[i,donors[i]]
                primal[cells[i]]=code
            primal_gain+=weight
            blocked.update(footprint|neighbors)
            cycles_seen+=1
        if primal_gain>best_gain:
            best,best_gain=primal,primal_gain
        if best_gain==upper:
            continue
        changed={int(cells[i]):i for i,j in enumerate(donors)
                 if i!=j or (i not in pinned and rotation[i,j]!=board[cells[i]]%3)}
        conflict=next(((i,changed[b]) for c,i in changed.items() for b in model.problem.neighbors[c]
                       if b in changed),None)
        if conflict is None:
            # Any zero-gain changes skipped above can be restored to the input.
            if best_gain!=upper:
                raise ArithmeticError('Unpriced independent donor cycle.')
            continue
        for i in conflict:
            child=tuple(sorted((*pinned,i)))
            if child not in visited:
                visited.add(child)
                heapq.heappush(queue,(-upper,child))
    if int(model.matches(best).sum()-model.matches(board).sum())!=best_gain:
        raise ArithmeticError('Full-board cycle price differs from the reconstructed score.')
    return best,dict(relaxations=solves,complete=not any(-upper>best_gain for upper,_ in queue),cycles=cycles_seen)

def diverse(model, rows, limit):
    entries=[]
    seen=set()
    for values in rows:
        board=model.legal_board(values)
        key=board.tobytes()
        if key in seen:
            continue
        seen.add(key)
        verified=result(model,board)
        entries.append((verified,physical_bonds(model,board)))
    entries.sort(key=lambda item:quality(item[0]),reverse=True)
    selected=[entries.pop(0)]
    while entries and len(selected)<limit:
        best=max(range(len(entries)),key=lambda i:min(len(entries[i][1]^old) for _,old in selected))
        selected.append(entries.pop(best))
    return [r for r,_ in selected]


def forest_pool(model,board,available,size,rng):
    """Build a forest of seam interactions, giving defective seams first choice."""
    allowed=set(map(int,available))
    if not allowed:
        return np.zeros(0,dtype=int)
    incident=np.zeros(model.n,dtype=int)
    failed=~model.matches(board)
    np.add.at(incident,model.a[failed],1)
    np.add.at(incident,model.b[failed],1)
    tie={int(c):i for i,c in enumerate(rng.permutation(sorted(allowed)))}
    selected=[]
    contacts={c:0 for c in allowed}
    while allowed and len(selected)<size:
        options=[c for c in allowed if contacts[c]<=1]
        if not options:
            break
        cell=max(options,key=lambda c:(contacts[c],int(incident[c]),tie[c]))
        selected.append(cell)
        allowed.remove(cell)
        for nbr in model.problem.neighbors[cell]:
            if nbr in allowed:
                contacts[int(nbr)]+=1
    return np.array(selected,dtype=int)


class Engine:
    def __init__(self,problem,*,bonds=False,joint=False,adjacent=False,ties=0,fork=False,
                 relink=False,slate=1,pool=10,tie_slack=0,cycle_budget=0,contacts=None,disjoint=False,exact_packing=False,pair_cycles=0,forest=False,contact_group=0,
                 touch_packing=None,separator_packing=False):
        if min(slate,pool)<1 or min(ties,cycle_budget)<0 or tie_slack not in (0,1,2):
            raise ValueError("Invalid candidate or repair limits.")
        self.model=BoardModel(problem)
        self.assemblies=Assemblies(self.model) if bonds else None
        if contacts is not None and (type(contacts) is not int or contacts < -1):
            raise ValueError("Invalid joint-repair contact limit.")
        if pair_cycles not in (0,2,3):
            raise ValueError("Expected zero, two or three for local cycle pairs.")
        if contact_group not in (0,1,2,3):
            raise ValueError("Contact groups must contain one, two or three cycles.")
        if contact_group and not pair_cycles:
            pair_cycles=3
        self.contact_group=contact_group
        self.forest=forest
        self.pair_cycles=pair_cycles
        if touch_packing not in (None, 'priced', 'complete', 'cp'):
            raise ValueError("Unknown fixed-orientation swap packing policy.")
        self.touch_packing = touch_packing
        self.separator_packing = separator_packing
        self.exact_packing = bool(exact_packing or touch_packing or separator_packing)
        self.contacts=contacts
        self.disjoint=disjoint
        self.joint=joint or adjacent or cycle_budget>0 or contacts is not None or disjoint or self.exact_packing or pair_cycles>0 or forest
        self.cycle_budget=cycle_budget
        self.adjacent=adjacent
        self.ties=ties
        self.fork=fork
        self.relink=relink
        self.slate=slate
        self.pool=pool
        self.tie_slack=tie_slack

    def repair(self, board, rng, step, stats, prices):
        model=self.model
        initial_board=board
        if self.assemblies:
            board=self.assemblies.relocate(board,rng)
            protected=self.assemblies.protected(board)
        else:
            protected=[]
        available=np.setdiff1d(model.free,protected)
        if self.joint and len(available):
            board,n=simultaneous_swaps(model,board,available,prices,exact=self.exact_packing,
                                       touch=self.touch_packing,packing_work=stats,
                                       separator_depth=3 if self.separator_packing else 0)
            stats['positive_swap_pairs']+=n
            stats['potential_pairs_priced']+=len(available)*(len(available)-1)//2
            if self.cycle_budget:
                board,cycle_work=pack_cycles(model,board,available,prices,relaxations=self.cycle_budget)
                for key,value in cycle_work.items():
                    stats['full_cycle_'+key]+=value
            # A capped model also explores longer donor cycles inside this variable pool.
            if self.forest:
                chosen=forest_pool(model,board,available,self.pool,rng)
            else:
                chosen=rng.choice(available,min(len(available),self.pool),replace=False)
            if self.pair_cycles:
                from r34_pair_cycles import pair_repair
                board,pair_work=pair_repair(model,board,chosen,self.pair_cycles,contact_group=self.contact_group)
                stats.update(pair_work)
            board,info=joint_repair(model,board,chosen,adjacent=self.adjacent,
                                   contacts=self.contacts,disjoint=self.disjoint)
            stats['joint_'+info['status']]+=1
            stats['joint_branches']+=info['branches']
        cells=np.setdiff1d(model.independent_set(board,rng),protected)
        if self.ties and len(cells):
            chosen=rng.choice(cells,min(self.pool,len(cells)),replace=False)
            other=np.setdiff1d(cells,chosen)
            board=model.block(board,other,rng)
            board,count=tie_repair(model,board,chosen,slack=self.tie_slack,cap=self.ties)
            stats['ties_validated']+=count
        else:
            board=model.block(board,cells,rng)
        board,rotation=model.orientations(board,rng=rng,movable=set(map(int,available)))
        if self.assemblies and self.assemblies.forbidden:
            board=self.assemblies.clear_forbidden(board,rng)
            if board is None:
                board=initial_board
                stats['forbidden_rejections']+=1
        stats['orientation_entries']+=rotation['table_entries']
        stats['block_assignments']+=int(len(cells)>0)
        if self.assemblies and not self.assemblies.bonds<=physical_bonds(model,board):
            raise ArithmeticError("Repair broke a mandatory physical bond.")
        stats['repair_steps']+=1
        return result(model,board)

    def _archive(self,seed,backend,replicas,dispatches,sampler_cache):
        if backend=='cpu':
            return [],0,0
        if sampler_cache:
            from r31_sampling import sample_archive
            r=sample_archive(self.model.problem,sampler_cache,seed=seed,backend=backend,
                             replicas=replicas,dispatches=dispatches)
            records=r['rows']
            solution=r['solution']
            proposals=r['move_proposals']
            new=r['new_proposals']
        else:
            from gpu_sampling_resume_round25 import SamplerSession
            session=SamplerSession(self.model.problem,seed=seed,backend=backend,replicas=replicas,
                                   target='gold',experiments={'replica_exchange':True,'move_tabu':True})
            try:
                while session.launches<dispatches and not session.done:
                    session.advance(60,max_dispatches=dispatches-session.launches)
                session.finish_phase()
                r=session.report()
                records=r['top_samples']
                solution=session.solution
                new=proposals=r['move_proposals']
            finally:
                session.close()
        rows=[r['codes'] for r in records if not r['domain_violations']]
        if solution is not None:
            rows.insert(0,solution)
        return rows,proposals,new

    def run(self,seed,*,backend='cpu',replicas=512,dispatches=64,rounds=32,sampler_cache=None,ranked=False,
            saved=None,save=None,batch=8,hints=(),trajectory_db=None):
        if batch<1 or seed<0 or dispatches<0 or rounds<0 or not 1<=replicas<=131072 or backend not in ('cpu','cuda','webgpu'):
            raise ValueError('Invalid independent search configuration.')
        model=self.model
        rng=np.random.default_rng(seed)
        from contextlib import closing
        import sqlite3
        # Single-path policies do not inspect the requested horizon when choosing moves.
        reusable=trajectory_db is not None and self.slate==1 and not self.fork and not self.relink
        if reusable:
            config=dict(bonds=self.assemblies is not None,joint=self.joint,adjacent=self.adjacent,
                        cycle_budget=self.cycle_budget,pool=self.pool,ties=self.ties,tie_slack=self.tie_slack)
            if self.forest:
                config['forest']=True
            if self.pair_cycles:
                config['pair_cycles']=self.pair_cycles
            if self.contact_group:
                config['contact_group']=self.contact_group
            if self.exact_packing:
                config['exact_packing']=True
            if self.touch_packing:
                config['touch_packing']=self.touch_packing
            if self.separator_packing:
                config['separator_packing']=True
            if self.contacts is not None or self.disjoint:
                config.update(contacts=self.contacts,disjoint=self.disjoint)
            key=[2 if self.assemblies else 1,domain_binding(model),config,seed,backend,replicas,dispatches,ranked,list(hints)]
            scope=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
            with closing(sqlite3.connect(trajectory_db,timeout=60)) as db,db:
                db.execute('CREATE TABLE IF NOT EXISTS r31_repair_prefix (scope TEXT, rounds INTEGER, gold INTEGER, '
                           'state TEXT, PRIMARY KEY(scope, rounds))')
                row=db.execute('SELECT state FROM r31_repair_prefix WHERE scope=? AND (rounds<=? OR gold=1) '
                               'ORDER BY gold DESC, rounds DESC LIMIT 1',(scope,rounds)).fetchone()
            if row:
                shared=json.loads(row[0])
                if saved is None or shared['best']['gold'] or shared['cursor']>saved['cursor']:
                    saved=shared
        started_at=saved['cursor'] if saved is not None else 0
        def keep_prefix(cursor,state):
            if reusable:
                with closing(sqlite3.connect(trajectory_db,timeout=60)) as db,db:
                    db.execute('INSERT OR IGNORE INTO r31_repair_prefix VALUES (?,?,?,?)',
                               (scope,cursor,int(state['best']['gold']),json.dumps(state)))
        if saved is None:
            if ranked:
                from r31_sampling import ranked_start
                first=ranked_start(model,seed)
            else:
                first=fresh_start(model,rng)
            archive,total,new=self._archive(seed,backend,replicas,dispatches,sampler_cache)
            rows=[first,*archive,*hints]
            if backend=='cpu':
                rows.extend(fresh_start(model,rng) for _ in range(self.slate-1))
            stats=Counter(gpu_proposals=total,new_gpu_proposals=new,repair_steps=0)
            starters=diverse(model,rows,self.slate)
            if starters[0]['gold']:
                return dict(**starters[0],seed=seed,initial=starters[0]['score'],work=dict(stats))
            states=[]
            best=None
            for start in starters:
                if self.assemblies:
                    candidate=self.assemblies.seed(start['codes'],rng)
                    start=result(model,candidate)
                states.append(dict(current=start,best=start,rng=np.random.default_rng(rng.integers(2**63)),
                                   prices=AssignmentPrices(model)))
                if best is None or quality(start)>quality(best):
                    best=start
            relinked=False
            initial=best['score']
            tested=set()
            trial_rounds=max(1,min(4,rounds//max(1,2*len(states))))
            forked=False
            cursor=0
        else:
            initial, best, forked, cursor=(saved[k] for k in ('initial','best','forked','cursor'))
            stats=Counter(saved['work'])
            tested=set(saved['tested'])
            rng.bit_generator.state=saved['rng']
            states=[]
            for row in saved['states']:
                child=np.random.default_rng()
                child.bit_generator.state=row['rng']
                states.append(dict(current=row['current'],best=row['best'],rng=child,prices=AssignmentPrices(model)))
            trial_rounds=saved['trial_rounds']
            relinked=saved.get('relinked',False)
        def checkpoint(cursor):
            if save or reusable:
                state=dict(version=1,initial=initial,best=best,forked=forked,cursor=cursor,
                          work=dict(stats),tested=sorted(tested),rng=rng.bit_generator.state,relinked=relinked,
                          trial_rounds=trial_rounds,states=[dict(current=s['current'],best=s['best'],
                              rng=s['rng'].bit_generator.state) for s in states])
                if save:
                    save(state)
                keep_prefix(cursor,state)
        if saved is None:
            checkpoint(0)
        for step in range(cursor,rounds):
            if best['gold']:
                break
            if self.relink and not relinked and len(states)>1 and step>=max(2,len(states)*trial_rounds):
                # Each parent received an equal, charged exploratory repair budget.
                parents=sorted(states,key=lambda s:quality(s['best']),reverse=True)
                paths=relink(model,parents[0]['best']['codes'],parents[1]['best']['codes'],self.assemblies)
                relinked=True
                stats['internal_candidates']+=len(paths)
                if paths:
                    mid=max((result(model,x) for x in paths),key=quality)
                    states.append(dict(current=mid,best=mid,rng=np.random.default_rng(rng.integers(2**63)),
                                       prices=AssignmentPrices(model)))
                    best=max((best,mid),key=quality)
                    index=len(states)-1
                else:
                    index=max(range(len(states)),key=lambda i:quality(states[i]['best']))
            elif self.fork and not forked and step>=rounds//2:
                parent=max(states,key=lambda s:quality(s['best']))
                forked=True
                if parent['best']['codes']!=parent['current']['codes']:
                    states.append(dict(current=parent['best'],best=parent['best'],
                                       rng=np.random.default_rng(rng.integers(2**63)),prices=AssignmentPrices(model)))
                    stats['forks']+=1
                    index=len(states)-1
                else:
                    index=states.index(parent)
            elif step<len(states)*trial_rounds or step%5==4:
                index=step%len(states)
            else:
                index=max(range(len(states)),key=lambda i:quality(states[i]['best']))
            if best['gold']:
                break
            state=states[index]
            candidate=self.repair(np.array(state['current']['codes']),state['rng'],step,stats,state['prices'])
            state['current']=candidate
            if quality(candidate)>quality(state['best']):
                state['best']=candidate
            if quality(candidate)>quality(best):
                best=candidate
            for threshold in (150,160,170,175,180,185,190,200,208):
                if best['score']>=threshold:
                    stats.setdefault(f'round_to_{threshold}',step+1)
            tested.add(hashlib.sha256(np.array(candidate['codes'],dtype='<u4').tobytes()).hexdigest())
            if (step+1)%batch==0 or step+1==rounds or best['gold']:
                checkpoint(step+1)
        stats['unique_repaired_boards']=len(tested)
        if self.assemblies:
            stats['mandatory_bonds']=len(self.assemblies.bonds)
        item=dict(**best,seed=seed,initial=initial,work=dict(stats))
        if trajectory_db:
            item['reused_repair_steps']=started_at
            item['new_repair_steps']=stats['repair_steps']-started_at
        return item


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--starts',type=int,default=1)
    parser.add_argument('--rounds',type=int,default=32)
    parser.add_argument('--backend',choices=('cpu','cuda','webgpu'),default='cpu')
    parser.add_argument('--dispatches',type=int,default=64)
    parser.add_argument('--replicas',type=int,default=512)
    parser.add_argument('--slate',type=int,default=1)
    parser.add_argument('--pool',type=int,default=10)
    parser.add_argument('--cycle-budget',type=int,default=0,help='Full-board assignment relaxations per repair.')
    parser.add_argument('--contacts',type=int,help='Maximum contacting changed pairs in joint repair (-1 for unlimited).')
    parser.add_argument('--forest',action='store_true',help='Choose a low-width repair pool along defective seams.')
    parser.add_argument('--pair-cycles',type=int,choices=(0,2,3),default=0,help='Price interacting donor cycles in the joint pool up to this length.')
    parser.add_argument('--contact-group',type=int,choices=(1,2,3),default=0,help='Pack positive cycles alone (1), or also price two or three touching cycles. With no --pair-cycles, consider cycles up to length three.')
    parser.add_argument('--exact-packing',action='store_true',help='Solve small and bipartite swap conflicts exactly.')
    parser.add_argument('--touch-packing',choices=('priced','complete','cp'),
                        help='Allow swaps on neighboring cells and price their exact oriented seam interactions. '
                             'Complete solves small components; cp also tries bounded fixed-code CP-SAT.')
    parser.add_argument('--separator-packing',action='store_true',
                        help='Split bounded nonbipartite components in the noncontacting swap pack.')
    parser.add_argument('--disjoint-contacts',action='store_true',help='Permit at most one changed neighbor per changed cell.')
    parser.add_argument('--ties',type=int,default=0)
    parser.add_argument('--tie-slack',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--sampler-cache',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--hint',type=Path,action='append',default=[],help='Optional full legal candidate boards.')
    for option in ('bonds','joint','adjacent','fork','relink','ranked'):
        parser.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    from r31_sampling import run_seeds
    p=SearchProblem(read_json(args.tiles))
    config={k:getattr(args,k) for k in ('bonds','joint','adjacent','ties','tie_slack','fork','relink','slate','pool','cycle_budget')}
    if args.forest:
        config['forest']=True
    if args.pair_cycles:
        config['pair_cycles']=args.pair_cycles
    if args.contact_group:
        config['contact_group']=args.contact_group
    if args.exact_packing or args.touch_packing or args.separator_packing:
        config['exact_packing']=True
    if args.touch_packing:
        config['touch_packing']=args.touch_packing
    if args.separator_packing:
        config['separator_packing']=True
    if args.contacts is not None or args.disjoint_contacts:
        config.update(contacts=args.contacts,disjoint=args.disjoint_contacts)
    engine=Engine(p,**config)
    experiment={k:getattr(args,k) for k in ('backend','dispatches','replicas','rounds','ranked')}
    hints=[engine.model.legal_board(read_json(path)['codes']).tolist() for path in args.hint]
    scope=hashlib.sha256(json.dumps([2 if args.bonds else 1,domain_binding(engine.model),config,experiment,hints],sort_keys=True).encode()).hexdigest()
    args.database.parent.mkdir(parents=True,exist_ok=True)
    def run(seed,saved,save):
        return engine.run(seed,**experiment,sampler_cache=args.sampler_cache,saved=saved,save=save,hints=hints,trajectory_db=args.database)
    def progress(row):
        print(json.dumps({k:v for k,v in row.items() if k!='codes'}),flush=True)
    report=run_seeds(args.database,scope,args.seed,args.starts,run,progress=progress,checkpointed=True,output=args.output)
    print(json.dumps(report))


if __name__=='__main__':
    main()
