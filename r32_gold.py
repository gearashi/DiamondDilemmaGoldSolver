"""Check Gold-specific physical facts, perform resumable loss searches and transport checked conflicts."""
import argparse
from contextlib import closing, ExitStack
import json
from math import comb
from pathlib import Path
import sqlite3
from time import monotonic

from r32_gold_bonds import bounded_loop_cores, inventory
from r32_gold_loss import (Certificates, digest, dropout, first_loss_partition, geographic,
                           master, pose_lower, read_json, reference_groups)
from r32_gold_partition import maximize_disjoint
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def _advance_saved(db, scopes, errors, start, covered=None):
    """Read an adjacent run of committed half-open intervals up to its first gap."""
    current = start if covered is None else covered
    placeholders = ','.join('?' for _ in scopes)
    rows = db.execute(f'SELECT lo,hi FROM r32_loss_pages WHERE scope IN ({placeholders}) '
                      'AND errors=? AND lo>=? ORDER BY lo', (*scopes, errors, start))
    with closing(rows):
        for lo, hi in rows:
            if lo > current:
                break
            current = max(current, hi)
    return current


def _higher_loss_scopes(db, identity, upper, reference_size, refined):
    """Find extant negative-page scopes from weaker models on the same coordinates."""
    if db.execute("SELECT 1 FROM r32_loss_pages LIMIT 1").fetchone() is None:
        return ()
    # Pre-v1.15 scopes had only an opaque digest. Every physically distinct
    # useful loss cap is bounded by the number of reference bonds.
    caps = set(range(upper+1,reference_size+1))
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='r34_loss_caps'").fetchone():
        caps.update(row[0] for row in db.execute(
            "SELECT cap FROM r34_loss_caps WHERE identity=? AND cap>?",
            (digest(identity),upper)))
    candidates = []
    for cap in sorted(caps):
        plain = digest([*identity,cap])
        candidates.append(plain)
        if refined:
            candidates.append(digest([plain,'integer-pattern-v1']))
    found = []
    for i in range(0,len(candidates),128):
        batch = candidates[i:i+128]
        values = ",".join(["(?)"]*len(batch))
        found.extend(row[0] for row in db.execute(
            f"WITH wanted(scope) AS (VALUES {values}) SELECT scope FROM wanted "
            "WHERE EXISTS (SELECT 1 FROM r32_loss_pages p WHERE p.scope=wanted.scope)",batch))
    return tuple(found)


def _negative_page(db, scope, errors, start, stop):
    """Fill only gaps if a legacy writer committed overlapping proof during compute."""
    inserted = 0
    previous = db.execute("SELECT hi FROM r32_loss_pages WHERE scope=? AND errors=? AND lo<=? "
                          "ORDER BY lo DESC LIMIT 1",(scope,errors,start)).fetchone()
    if previous:
        start = max(start,previous[0])
    rows = list(db.execute("SELECT lo,hi FROM r32_loss_pages WHERE scope=? AND errors=? "
                           "AND lo>=? AND lo<? ORDER BY lo",(scope,errors,start,stop)))
    for lo,hi in rows:
        if start < lo:
            db.execute("INSERT INTO r32_loss_pages VALUES(?,?,?,?)",(scope,errors,start,lo))
            inserted += 1
        start = max(start,hi)
    if start<stop:
        db.execute("INSERT INTO r32_loss_pages VALUES(?,?,?,?)",(scope,errors,start,stop))
        inserted += 1
    return inserted


def run_syndrome(problem, ref, groups, upper, indices, db_path, seconds, page=2048, emit=None, *, refine_integer=False, all_groups=False, sphere_euler=False, mate_support=False, mate_cost=False, geographic_hints=(), contact_hints=()):
    """Replay optional geometric evidence before handing it to the ranked loss search."""
    from r35_geographic_replay import geographic_departures
    ref=frozenset(ref)
    geo=geographic_departures(problem,ref,inventory(problem).required,geographic_hints) if geographic_hints else ()
    from r35_induced_contacts import contact_departures
    contact=contact_departures(problem,ref,inventory(problem).required,contact_hints) if contact_hints else ()
    result=_ranked_loss(problem,ref,groups,upper,indices,db_path,seconds,page,emit,refine_integer,
                        all_groups,sphere_euler,mate_support,mate_cost,tuple(dict.fromkeys((*geo,*contact))),contacts_present=bool(contact))
    if geo:
        result.update(verified_geographic_departures=len(geo),
                      relaxation=('signed_inventory_with_geographic_departures' if refine_integer
                                  else 'parity_with_geographic_departures'))
    if contact:
        result.update(verified_edge_perfect_departures=len(contact),
                      relaxation=('signed_inventory_with_edge_perfect_departures' if refine_integer
                                  else 'parity_with_edge_perfect_departures'))
    return result


def _ranked_loss(problem,ref,groups,upper,indices,db_path,seconds,page,emit,refine_integer,
                 all_groups,sphere_euler,mate_support,mate_cost,geo,*,contacts_present=False):
    """Save completed disjoint prefix ranges together with their negative result."""
    if page<1 or seconds<0:raise ValueError('Expected positive page size and nonnegative duration')
    ref = frozenset(ref)
    groups = [frozenset(g) for g in groups]
    layout = OutsideSyndromes.layout(ref, groups, inventory(problem).required)
    packed, required, outside = layout
    max_error = min(len(outside), max(0, upper-len(packed)))
    all_layers_requested = indices is None
    indices = list(range(max_error+1)) if all_layers_requested else list(indices)
    identity = [3,problem.fingerprint,sorted(ref),sorted(required),[sorted(c) for c in packed]]
    packed_scope = digest([*identity,upper])
    packed_identity = digest(identity)
    aggregate=all_groups and any(g not in packed for g in groups)
    charged_groups=groups if aggregate else packed
    if aggregate:
        identity=[*identity,'all-verified-groups-v1',sorted(sorted(g) for g in groups)]
    weaker_scope=digest([*identity,upper])
    cheaper_price_scopes=[]
    conditions=None
    if sphere_euler or mate_support or mate_cost:
        if not refine_integer:raise ValueError('Extra signed physical conditions require integer refinement.')
        from r35_signed_conditions import SignedConditions
        conditions=SignedConditions(problem,euler=sphere_euler,mates=mate_support,mate_cost=mate_cost)
        identity=[*identity,'signed-physical-conditions-v1',sphere_euler,mate_support or mate_cost]
        if mate_cost:
            cheaper_price_scopes.extend((identity,[*identity,'eight-side-loss-prices-v1']))
            identity=[*identity,'small-family-loss-prices-v2']
    scope = digest([*identity,upper])
    # A plain negative page excludes every integer-feasible pattern inside it.
    # A plain hit can coexist with further candidates in the same prefix.
    plain_scope = scope
    if refine_integer:
        scope = digest([plain_scope, 'integer-pattern-v1'])
    read_scopes = (plain_scope, scope) if refine_integer else (scope,)
    if conditions is not None:
        read_scopes += (weaker_scope,digest([weaker_scope,'integer-pattern-v1']))
    if aggregate:
        read_scopes += (packed_scope,)
        if refine_integer:
            read_scopes += (digest([packed_scope,'integer-pattern-v1']),)
    weaker_identity=None
    # The packed coordinates and outside ordering remain the original ones.
    # Only negatives transport from the corresponding weaker physical scope.
    # Any phase from that scope is independently repriced with the added groups.
    if geo:
        weaker_identity=identity
        identity=[*identity,'verified-geographic-departures-v1',sorted(sorted(g) for g in geo)]
        charged_groups=[*charged_groups,*geo]
        geo_plain=digest([*identity,upper])
        scope=digest([geo_plain,'integer-pattern-v1']) if refine_integer else geo_plain
        read_scopes=tuple(dict.fromkeys((scope,geo_plain,*read_scopes)))
    if contacts_present:
        from r35_contact_certificate import checked_contact_bound
        contact_bound=checked_contact_bound(problem,ref,required,groups,charged=charged_groups)
        if upper<contact_bound:
            return dict(status='INFEASIBLE',loss_budget=upper,checked_layers=indices,
                        all_outside_layers=all_layers_requested,certified_loss_lower_bound=contact_bound,
                        work={'decisions':0,'contact_weight_certificate':1})
    if cheaper_price_scopes:
        if geo:
            geographic_tag=['verified-geographic-departures-v1',sorted(sorted(g) for g in geo)]
            cheaper_price_scopes.extend([*prior,*geographic_tag] for prior in tuple(cheaper_price_scopes))
        for cheaper in cheaper_price_scopes:
            weak=digest([*cheaper,upper])
            read_scopes+=tuple(v for v in (weak,digest([weak,'integer-pattern-v1'])) if v not in read_scopes)
    searched_caps = registered_cap = False
    budget = monotonic()+seconds if seconds else float('inf')
    pattern_filter = None
    excludes = None
    saved_hints = None

    def get_filter():
        nonlocal pattern_filter, excludes
        if pattern_filter is None:
            if aggregate or conditions is not None or geo:
                from r35_overlap_loss import OverlapPattern
                pattern_filter=OverlapPattern(problem,packed,charged_groups,required,outside,upper,refine_integer,work,conditions)
            else:
                from r33_gold_refine import IntegerCoreStore, make_integer_filter
                pattern_filter = make_integer_filter(problem,ref,packed,required,outside,upper,
                                                      store=IntegerCoreStore(db,scope),work=work)
            close_filter = getattr(pattern_filter,"close",None)
            if close_filter is not None:
                renewal.callback(close_filter)
            excludes = getattr(pattern_filter,"excludes",None)
        return pattern_filter

    def accept_errors(indices, deadline):
        return get_filter()(indices,deadline)

    def reject_before_joint(indices, deadline):
        nonlocal saved_hints
        if saved_hints is None:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='r34_integer_cores' AND type='table'").fetchone()
            saved_hints = bool(exists and db.execute("SELECT 1 FROM r34_integer_cores WHERE scope=? LIMIT 1",(scope,)).fetchone())
            if saved_hints:
                get_filter()
        return excludes is not None and excludes(indices,deadline)

    work = {}
    total = solved = pages = 0
    witness_hits = None
    identity_key = digest(identity)
    from r35_loss_bound import checked_lower_bound, REFERENCE_PHASES, published_hit
    if any(type(e) is not int or not 0 <= e <= len(outside) for e in indices) or type(upper) is not int or upper < 0:
        raise ValueError('Invalid loss cap or outside-error layer')
    if geo and not all(geo):
        covered=sum(comb(len(outside)-min(e,2),max(0,e-2)) for e in indices)
        return dict(status='INFEASIBLE',loss_budget=upper,checked_layers=indices,
                    all_outside_layers=set(indices)>=set(range(max_error+1)),completed_prefixes=covered,
                    total_prefixes=covered,new_pages=0,work={'required_geographic_conflict':1})
    bound=checked_lower_bound(packed_identity,charged_groups) if refine_integer else None
    if bound is not None and upper < bound:
        covered = sum(comb(len(outside)-min(e,2),max(0,e-2)) for e in indices)
        return dict(status='INFEASIBLE',loss_budget=upper,checked_layers=indices,
                    all_outside_layers=set(indices)>=set(range(max_error+1)),completed_prefixes=covered,
                    total_prefixes=covered,new_pages=0,work={'reference_loss_certificate_reused':1},
                    certified_loss_lower_bound=bound,
                    relaxation="all_group_signed_inventory" if aggregate else "disjoint_group_signed_inventory")
    if conditions is not None:
        conditions=conditions.for_loss(ref,charged_groups,required,upper)
    public_hits={}
    for phase in REFERENCE_PHASES:
        if conditions is not None and not conditions.accepts(phase):continue
        positive=published_hit(problem,ref,charged_groups,required,outside,packed_identity,upper,refine_integer,phase)
        if positive is not None:
            if conditions is not None:positive['loss_cost']=conditions.price(phase,positive['loss_cost'])
            if aggregate:
                positive['relaxation']='all_group_signed_inventory' if refine_integer else 'all_group_parity'
            old=public_hits.get(positive['errors'])
            if old is None or positive['loss_cost']<old['loss_cost']:
                public_hits[positive['errors']]=positive
    if all_layers_requested and public_hits:
        return dict(min(public_hits.values(),key=lambda row:row['loss_cost']),pages=0,work={})
    query = None
    from r34_rank_leases import RankedLeases, LostRankLease
    with closing(sqlite3.connect(db_path, timeout=30, uri=True, isolation_level=None)) as db, ExitStack() as renewal:
        schema = db.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN ('r32_loss_pages','r32_phase_hits')").fetchone()[0]
        if schema != 2:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS r32_loss_pages(scope TEXT, errors INTEGER,lo INTEGER,hi INTEGER, PRIMARY KEY(scope,errors,lo))')
                db.execute('CREATE TABLE IF NOT EXISTS r32_phase_hits(scope TEXT, errors INTEGER, result TEXT, PRIMARY KEY(scope,errors))')
        hit_sql = 'SELECT result FROM r32_phase_hits WHERE scope=? AND errors=?'
        for errors in indices:
            if errors<0 or errors>len(outside):raise ValueError('Outside-error layer out of range')
            count = comb(len(outside)-min(errors,2),max(0,errors-2))
            total += count
            known = db.execute(hit_sql, (scope,errors)).fetchone()
            if known:
                return dict(json.loads(known[0]), reused=True, pages=pages, work=work)
            if witness_hits is None:
                from r35_loss_witness import checked_hits
                witness_hits = checked_hits(db,identity_key,problem,ref,charged_groups,required,outside,upper,refine_integer,conditions)
                for cheaper in ([weaker_identity] if weaker_identity is not None else [])+cheaper_price_scopes:
                    weak_hits=checked_hits(db,digest(cheaper),problem,ref,charged_groups,
                                           required,outside,upper,refine_integer,conditions)
                    for layer,positive in weak_hits.items():
                        witness_hits.setdefault(layer,positive)
                for layer,positive in public_hits.items():
                    witness_hits.setdefault(layer,positive)
            if errors in witness_hits:
                return dict(witness_hits[errors],pages=pages,work=work)
            current = _advance_saved(db, read_scopes, errors, 0)
            if current < count and not searched_caps:
                higher = _higher_loss_scopes(db, identity, upper, len(ref), refine_integer)
                for cheaper in ([weaker_identity] if weaker_identity is not None else [])+cheaper_price_scopes:
                    higher += _higher_loss_scopes(db,cheaper,upper,len(ref),refine_integer)
                searched_caps = True
                if higher:
                    read_scopes += higher
                    current = _advance_saved(db, read_scopes, errors, 0)
            leases = None
            busy = False
            while current < count or busy:
                if current>=count:
                    current=_advance_saved(db,read_scopes,errors,0)
                    if current>=count:
                        break
                    known=db.execute(hit_sql,(scope,errors)).fetchone()
                    if known:return dict(json.loads(known[0]),reused=True,pages=pages,work=work)
                    leases.wait(current,budget,ready=lambda:db.execute(hit_sql,(scope,errors)).fetchone())
                    busy=False
                    continue
                if query is None:
                    query = OutsideSyndromes(problem, ref, groups, required, layout=layout)
                with db:
                    db.execute('BEGIN IMMEDIATE')
                    known=db.execute(hit_sql,(scope,errors)).fetchone()
                    if known:return dict(json.loads(known[0]),reused=True,pages=pages,work=work)
                    covered = current
                    for source in read_scopes:
                        row = db.execute('SELECT hi FROM r32_loss_pages WHERE scope=? AND errors=? '
                                         'AND lo<=? ORDER BY lo DESC LIMIT 1', (source,errors,current)).fetchone()
                        if row:
                            covered = max(covered,row[0])
                    if covered > current:
                        current = _advance_saved(db,read_scopes,errors,current,covered)
                        continue
                    remaining=budget-monotonic()
                    if remaining<=0:return dict(status='UNKNOWN',errors=errors,next_prefix=current,pages=pages,work=work)
                    end = min(count,current+page)
                    for source in read_scopes:
                        other = db.execute('SELECT min(lo) FROM r32_loss_pages WHERE scope=? AND errors=? AND lo>?',
                                           (source, errors, current)).fetchone()[0]
                        if other is not None:
                            end = min(end,other)
                    if leases is None:
                        leases = RankedLeases(db,'gold-loss',scope+'/'+str(errors),private=db_path==':memory:',lifetime=renewal)
                        db.execute("CREATE TABLE IF NOT EXISTS r34_loss_cursors(scope TEXT,errors INTEGER,"
                                   "prefix INTEGER,cursor TEXT,PRIMARY KEY(scope,errors,prefix))")
                    claim,skip = leases.take(current,end)
                    if claim is None:
                        current=skip
                        busy=True
                        continue
                    end = claim.hi
                    previous=db.execute("SELECT cursor FROM r34_loss_cursors WHERE scope=? AND errors=? AND prefix=?",
                                        (scope,errors,current)).fetchone()
                    try:
                        query.resume_cursor=json.loads(previous[0]) if previous else None
                    except (ValueError,TypeError):
                        query.resume_cursor=None
                written=current
                def publish_partial(rank,cursor):
                    nonlocal written,pages,registered_cap
                    if not written<=rank<=end:
                        raise ValueError("Loss checkpoint escaped its owned interval")
                    if rank>written:
                        if upper>len(ref) and not registered_cap:
                            db.execute("CREATE TABLE IF NOT EXISTS r34_loss_caps(identity TEXT, cap INTEGER, "
                                       "PRIMARY KEY(identity,cap))")
                            db.execute("INSERT OR IGNORE INTO r34_loss_caps VALUES (?,?)",(digest(identity),upper))
                            registered_cap=True
                        pages+=_negative_page(db,scope,errors,written,rank)
                        db.execute("DELETE FROM r34_loss_cursors WHERE scope=? AND errors=? AND prefix>=? AND prefix<?",
                                   (scope,errors,written,rank))
                        written=rank
                    if cursor is not None:
                        db.execute("INSERT OR REPLACE INTO r34_loss_cursors VALUES(?,?,?,?)",
                                   (scope,errors,rank,json.dumps(cursor)))
                def checkpoint(rank,cursor):
                    with db:
                        db.execute("BEGIN IMMEDIATE")
                        leases.fence(claim)
                        publish_partial(rank,cursor)
                query.checkpoint_progress=checkpoint
                try:
                    with leases.computing(claim):
                        options = {'accept_errors':accept_errors, 'reject_before_joint':reject_before_joint,
                                   'verified_cores':lambda: getattr(pattern_filter,'verified',())} if refine_integer else {}
                        if aggregate or conditions is not None or geo:
                            options={'accept_errors':accept_errors,
                                     'accept_status':('INTEGER_RELAXATION_FEASIBLE' if refine_integer else 'PHASE_RELAXATION_FEASIBLE')}
                        out=query.indexed(upper,errors,remaining,start=current,stop=end,**options)
                        for k,v in out['work'].items():work[k]=work.get(k,0)+v
                        with db:
                            db.execute("BEGIN IMMEDIATE")
                            leases.fence(claim)
                            publish_partial(out['next_rank'],out.get('cursor'))
                            current=out['next_rank']
                            if out['status'] in ('PHASE_RELAXATION_FEASIBLE','INTEGER_RELAXATION_FEASIBLE'):
                                witness = getattr(pattern_filter,'last_witness',None)
                                if witness is not None and witness[1] is not None:
                                    from r35_loss_witness import save_witness
                                    save_witness(db,identity_key,witness[1])
                                db.execute('INSERT OR IGNORE INTO r32_phase_hits VALUES(?,?,?)',(scope,errors,json.dumps(out)))
                                saved=db.execute(hit_sql,(scope,errors)).fetchone()
                                return dict(json.loads(saved[0]),pages=pages,work=work)
                            if out['status']=='UNKNOWN':return dict(out,pages=pages,work=work)
                except LostRankLease:
                    busy=True
                    current=_advance_saved(db,read_scopes,errors,0)
                    continue
                finally:
                    query.checkpoint_progress=None
                    query.resume_cursor=None
                if emit:emit(errors,current,count)
            if leases is not None:
                leases.close()
            solved+=count
    return dict(status='INFEASIBLE',loss_budget=upper,checked_layers=list(indices),
                all_outside_layers=set(indices)>=set(range(max_error+1)),
                completed_prefixes=solved,total_prefixes=total,new_pages=pages,work=work)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inventory','loss','integer-bound','dropout','master','geographic','poses','cores','loops','watch'))
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--center',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--backend',choices=('cpu','cuda'),default='cpu',help='Runtime for the checked placement watcher')
    parser.add_argument('--replicas',type=int,default=128)
    parser.add_argument('--claims',action='append',default=[],help='JSON file with clauses to replay')
    parser.add_argument('--case',help='JSON with optional codes, pins, domains, required and forbidden')
    parser.add_argument('--database',default=':memory:')
    parser.add_argument('--seconds',type=float,default=15,help='Zero is unlimited in loss mode')
    parser.add_argument('--upper',type=int,default=17)
    parser.add_argument('--lower',type=int,default=0)
    parser.add_argument('--outside',type=int,help='In loss, outside phase errors. In geographic, exact physical bond losses outside the packed groups.')
    parser.add_argument('--page',type=int,default=8192)
    parser.add_argument('--refine-integer',action='store_true',help='In loss mode, check signed integer inventories before accepting a phase pattern (wide phase images use OR-Tools).')
    parser.add_argument('--independent-native',action='store_true',help='In integer-bound, rerun the native all-pattern solver even for an independently checked published result.')
    parser.add_argument('--sphere-euler',action='store_true',help='Require the spherical one-loop Euler equation in signed integer search.')
    parser.add_argument('--mate-cost',action='store_true',help='Price lost reference pairs in small reciprocal classes, extending to twelve sides for broad overlapping departures.')
    parser.add_argument('--mate-support',action='store_true',help='Require signed physical mates and perfect matchings in reciprocal classes of up to sixteen sides.')
    parser.add_argument('--excess-charges',action='store_true',help='In integer-bound, subtract disjoint compulsory departures before charging the remaining losses.')
    parser.add_argument('--all-groups',action='store_true',help='In loss or integer-bound, enforce every verified physical departure group, including overlaps.')
    parser.add_argument('--stage',choices=('xor','integer','euler','mating'),default='integer')
    parser.add_argument('--branch',nargs='*',type=int,default=[])
    parser.add_argument('--limit',type=int,default=10000)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.seconds<0 or args.limit<0 or args.page<1 or args.upper<0 or (args.mode not in ('loss','integer-bound') and args.seconds==0):
        parser.error('Invalid budget')
    if (args.sphere_euler or args.mate_support or args.mate_cost) and not (args.mode=='integer-bound' or args.mode=='loss' and args.refine_integer):
        parser.error('Signed physical conditions require integer-bound or loss --refine-integer')
    if args.independent_native and args.mode!='integer-bound':parser.error('--independent-native requires integer-bound')
    if args.excess_charges and args.mode!='integer-bound':parser.error('--excess-charges requires integer-bound')
    if args.all_groups and args.mode not in ('loss','integer-bound'):parser.error('--all-groups requires loss or integer-bound')
    if args.backend!='cpu' and args.mode!='watch':parser.error('CUDA applies to watch mode')
    if args.refine_integer:
        if args.mode != 'loss':
            parser.error('--refine-integer is available in loss mode')
    p=SearchProblem(read_json(args.tiles),physical_gold_support=True)
    case=read_json(args.case) if args.case else {}
    if 'domains' in case:
        from r29_gold_geography import checked_domains
        ds=checked_domains(p,[sum(1<<v for v in set(row)) if isinstance(row,list) else row
                              for row in case['domains']])
    else:ds=None
    rules=inventory(p)
    if args.mode=='inventory':
        out=dict(required=sorted(rules.required),forbidden=sorted(rules.forbidden),possible=len(rules.possible),
                 single_bond_loops=sorted(rules.loops))
    else:
        codes=case['codes'] if 'codes' in case else read_json(args.center)['codes']
        extra=[];geographic_hints=[];contact_hints=[]
        for file in args.claims:
            hints=read_json(file)
            extra.extend(hints.get('clauses',()))
            geographic_hints.extend(hints.get('geographic_cores',()))
            contact_hints.extend(hints.get('edge_perfect_cores',()))
        ref,cert,groups=reference_groups(p,codes,extra)
        for pair in [*case.get('required',[]),*case.get('forbidden',[])]:
            if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or
                    any(type(v) is not int or not 0<=v<3*p.n for v in pair) or pair[0]==pair[1]):
                raise ValueError("Invalid physical pair in case")
        requested={tuple(sorted(x)) for x in case.get('required',[])}
        if not requested<=rules.possible:
            raise ValueError("Required pair cannot participate in a Gold sewing")
        req=set(rules.required)|requested
        forbid=set(rules.forbidden)|{tuple(sorted(x)) for x in case.get('forbidden',[])}
        if args.mode=='loss':
            if case.keys()-{'codes'}:parser.error('loss accepts an unrestricted physical reference only')
            indices=None if args.outside is None else [args.outside]
            out=run_syndrome(p,ref,groups,args.upper,indices,args.database,args.seconds,args.page,
                             refine_integer=args.refine_integer,all_groups=args.all_groups,sphere_euler=args.sphere_euler,
                             mate_support=args.mate_support,mate_cost=args.mate_cost,geographic_hints=geographic_hints,contact_hints=contact_hints)
        elif args.mode=='integer-bound':
            if case.keys()-{'codes'} or args.outside is not None:
                parser.error('integer-bound solves every outside pattern of one unrestricted reference')
            from r35_loss_bound import global_integer
            packed, mandatory, outside = OutsideSyndromes.layout(ref, groups, rules.required)
            extra_groups = [g for g in groups if g not in packed] if args.all_groups else []
            from r35_geographic_replay import geographic_departures
            geographic_groups = geographic_departures(p, ref, mandatory, geographic_hints)
            from r35_induced_contacts import contact_departures
            contact_groups=contact_departures(p,ref,mandatory,contact_hints)
            conditional_groups=tuple(dict.fromkeys((*geographic_groups,*contact_groups)))
            contact_bound=0
            if contact_groups and not args.independent_native:
                from r35_contact_certificate import checked_contact_bound
                contact_bound=checked_contact_bound(p,ref,mandatory,groups,
                    charged=[*packed,*extra_groups,*conditional_groups])
            if contact_bound and args.upper<contact_bound:
                out=dict(status='INFEASIBLE',cap=args.upper,all_outside_layers=True,
                         certified_loss_lower_bound=contact_bound,
                         work={'decisions':0,'contact_weight_certificate':1})
            elif any(not group for group in conditional_groups):
                out = dict(status='INFEASIBLE', cap=args.upper, all_outside_layers=True,
                           work={'decisions': 0, 'required_geographic_conflict': 1})
            else:
                conditions=None
                if args.sphere_euler or args.mate_support or args.mate_cost:
                    from r35_signed_conditions import SignedConditions
                    conditions=SignedConditions(p,euler=args.sphere_euler,mates=args.mate_support,mate_cost=args.mate_cost)
                out = global_integer(p,ref,packed,mandatory,outside,args.upper,args.seconds,
                                         additional_groups=[*extra_groups, *conditional_groups],
                                         excess=args.excess_charges,independent_native=args.independent_native,conditions=conditions)
            if geographic_groups:
                out.update(relaxation='signed_inventory_with_geographic_departures',
                           verified_geographic_departures=len(geographic_groups))
            if contact_groups:
                out.update(relaxation='signed_inventory_with_edge_perfect_departures',
                           verified_edge_perfect_departures=len(contact_groups))
        elif args.mode=='master':
            out=master(p,ref,cert,args.upper,args.seconds,req,forbid,ds)
        elif args.mode=='geographic':
            if case.keys()-{'codes','domains','pins'}:parser.error('Use branch for geographic bond conditions')
            out=geographic(p,ref,maximize_disjoint(groups),cert.rows,args.branch,args.lower,args.upper,args.outside,
                           args.seconds,args.database,ds,{int(i):v for i,v in case.get('pins',{}).items()})
        elif args.mode=='dropout':
            out=dropout(p,ref,args.upper,groups,args.seconds,args.stage,req,forbid,ds)
        elif args.mode=='poses':
            out=pose_lower(p,ref,groups,codes,args.seconds,case.get('free_cells'))
        elif args.mode=='cores':
            out=dict(clauses=cert.clauses(),reference_groups=len(groups),packed=len(maximize_disjoint(groups)))
        elif args.mode=='watch':
            from constraint_dfs import solve
            p.physical_conflicts=cert.clauses()
            p.geographic_conflicts=geographic_hints
            if ds is not None:
                from r27_structure_filters import bits
                p.domains=[list(bits(row)) for row in ds]
            if args.backend=='cuda':
                from constraint_shared_conflicts import VerifiedConflictStore, MAX_LITERALS
                from gpu_domain_runner import solve as device_solve
                from r32_gold_watch import placed_bonds
                # Reconstruct the physical witness at this known placement; each
                # transported clause is checked again against original tile data.
                joins=placed_bonds(p,enumerate(codes))
                store=VerifiedConflictStore(p,'gold')
                admitted=0
                try:
                    for group in groups:
                        literals=sorted({literal for pair in group for literal in joins[pair]})
                        if (len(literals)<=MAX_LITERALS and
                                all(v in p.domains[c] for c,v in literals)):
                            admitted+=store.admit_physical_conflict(group,literals)
                    out=device_solve(p,target='gold',seconds=args.seconds,replicas=args.replicas,
                                     backend='cuda',options={'gpu_shared_clauses':True},
                                     learning_state={'_shared_conflicts':store})
                finally:
                    store.close()
                out['physical_clauses_admitted']=admitted
            else:
                out=solve(p,target='gold',seconds=args.seconds)
            out.pop('checkpoint',None)
        else:
            out_rows,visits=bounded_loop_cores(p,req,args.limit)
            cert.add(out_rows)
            out=dict(clauses=cert.clauses(),candidate_queries=visits)
    out['fingerprint']=p.fingerprint
    if args.sphere_euler or args.mate_support or args.mate_cost:
        out['signed_conditions']=dict(sphere_euler=args.sphere_euler,mate_support=args.mate_support or args.mate_cost)
        if args.mate_cost:out['signed_conditions']['mate_cost']=True
    message=json.dumps(out,indent=2)
    if args.output:Path(args.output).write_text(message+'\n',encoding='utf-8')
    print(message)


if __name__=='__main__':main()
