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


def run_syndrome(problem, ref, groups, upper, indices, db_path, seconds, page=2048, emit=None, *, refine_integer=False):
    """Save completed disjoint prefix ranges together with their negative result."""
    if page<1 or seconds<0:raise ValueError('Expected positive page size and nonnegative duration')
    layout = OutsideSyndromes.layout(ref, groups, inventory(problem).required)
    packed, required, outside = layout
    max_error = min(len(outside), max(0, upper-len(packed)))
    indices = list(range(max_error+1)) if indices is None else list(indices)
    identity = [3,problem.fingerprint,sorted(ref),sorted(required),[sorted(c) for c in packed]]
    scope = digest([*identity,upper])
    # A plain negative page excludes every integer-feasible pattern inside it.
    # A plain hit can coexist with further candidates in the same prefix.
    plain_scope = scope
    if refine_integer:
        scope = digest([plain_scope, 'integer-pattern-v1'])
    read_scopes = (plain_scope, scope) if refine_integer else (scope,)
    searched_caps = registered_cap = False
    budget = monotonic()+seconds if seconds else float('inf')
    pattern_filter = None
    excludes = None
    saved_hints = None

    def get_filter():
        nonlocal pattern_filter, excludes
        if pattern_filter is None:
            from r33_gold_refine import IntegerCoreStore, IntegerPatternFilter
            pattern_filter = IntegerPatternFilter(problem,ref,packed,required,outside,upper,
                                                  store=IntegerCoreStore(db,scope),work=work)
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
            current = _advance_saved(db, read_scopes, errors, 0)
            if current < count and not searched_caps:
                higher = _higher_loss_scopes(db, identity, upper, len(ref), refine_integer)
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
                        options = {'accept_errors':accept_errors, 'reject_before_joint':reject_before_joint} if refine_integer else {}
                        out=query.indexed(upper,errors,remaining,start=current,stop=end,**options)
                        for k,v in out['work'].items():work[k]=work.get(k,0)+v
                        with db:
                            db.execute("BEGIN IMMEDIATE")
                            leases.fence(claim)
                            publish_partial(out['next_rank'],out.get('cursor'))
                            current=out['next_rank']
                            if out['status'] in ('PHASE_RELAXATION_FEASIBLE','INTEGER_RELAXATION_FEASIBLE'):
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
    parser.add_argument('mode',choices=('inventory','loss','dropout','master','geographic','poses','cores','loops','watch'))
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
    parser.add_argument('--refine-integer',action='store_true',help='In loss mode, check signed integer inventories before accepting a phase pattern (requires OR-Tools).')
    parser.add_argument('--stage',choices=('xor','integer','euler','mating'),default='integer')
    parser.add_argument('--branch',nargs='*',type=int,default=[])
    parser.add_argument('--limit',type=int,default=10000)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.seconds<0 or args.limit<0 or args.page<1 or args.upper<0 or (args.mode!='loss' and args.seconds==0):
        parser.error('Invalid budget')
    if args.backend!='cpu' and args.mode!='watch':parser.error('CUDA applies to watch mode')
    if args.refine_integer:
        if args.mode != 'loss':
            parser.error('--refine-integer is available in loss mode')
        from importlib.util import find_spec
        if find_spec('ortools') is None:
            parser.error('--refine-integer requires OR-Tools from the full Python 3.12 setup')
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
        extra=[]
        for file in args.claims:extra.extend(read_json(file)['clauses'])
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
                             refine_integer=args.refine_integer)
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
    message=json.dumps(out,indent=2)
    if args.output:Path(args.output).write_text(message+'\n',encoding='utf-8')
    print(message)


if __name__=='__main__':main()
