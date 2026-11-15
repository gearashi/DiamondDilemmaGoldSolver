"""Incremental shortest physical routes as plateau score floors fall."""
import hashlib
import json
import time

from r28_exchange import ExchangeBound
from r35_plateau_pages import ExpansionStopped, advance, stored_vector


def schema(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS walk_vertex_labels(id INTEGER PRIMARY KEY, board TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS walk_scored_edges(
            board INTEGER, score INTEGER, neighbor INTEGER, PRIMARY KEY(board,score,neighbor)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS walk_scored_centers(board TEXT PRIMARY KEY);
        CREATE TRIGGER IF NOT EXISTS walk_index_new_edge AFTER INSERT ON walk_edges
            WHEN EXISTS(SELECT 1 FROM walk_scored_centers WHERE board=NEW.board)
        BEGIN
            INSERT OR IGNORE INTO walk_vertex_labels(board) VALUES(NEW.neighbor);
            INSERT OR IGNORE INTO walk_scored_edges
            SELECT a.id,coalesce(json_extract(o.result,'$.score'),l.score),b.id
            FROM walk_vertex_labels a JOIN walk_vertex_labels b ON b.board=NEW.neighbor
            LEFT JOIN optima o ON o.key=NEW.neighbor
            LEFT JOIN walk_legacy l ON l.board=NEW.neighbor
            WHERE a.board=NEW.board AND (o.key IS NOT NULL OR l.board IS NOT NULL);
        END;
        CREATE TABLE IF NOT EXISTS walk_widen(scope TEXT PRIMARY KEY, level INTEGER, done INTEGER);
        CREATE TABLE IF NOT EXISTS walk_distances(
            scope TEXT,board TEXT,score INTEGER,depth INTEGER,path TEXT,parent TEXT,
            dirty INTEGER,expanded_floor INTEGER,changed_path INTEGER,next_floor INTEGER,
            PRIMARY KEY(scope,board));
        CREATE INDEX IF NOT EXISTS walk_distance_queue ON walk_distances(scope,depth,path) WHERE dirty=1;
        CREATE INDEX IF NOT EXISTS walk_distance_goals ON walk_distances(scope,score,depth);
        CREATE INDEX IF NOT EXISTS walk_distance_activation ON walk_distances(scope,next_floor);
    """)


def index_center(db,key):
    fresh=db.execute('INSERT OR IGNORE INTO walk_scored_centers VALUES(?)',(key,)).rowcount
    if fresh:
        db.execute('INSERT OR IGNORE INTO walk_vertex_labels(board) VALUES(?)',(key,))
        db.execute('INSERT OR IGNORE INTO walk_vertex_labels(board) SELECT neighbor FROM walk_edges WHERE board=?',(key,))
    label=db.execute('SELECT id FROM walk_vertex_labels WHERE board=?',(key,)).fetchone()[0]
    if fresh:
        db.execute("""INSERT OR IGNORE INTO walk_scored_edges
            SELECT ?,coalesce(json_extract(o.result,'$.score'),l.score),v.id FROM walk_edges e
            JOIN walk_vertex_labels v ON v.board=e.neighbor
            LEFT JOIN optima o ON o.key=e.neighbor LEFT JOIN walk_legacy l ON l.board=e.neighbor
            WHERE e.board=? AND (o.key IS NOT NULL OR l.board IS NOT NULL)""",(label,key))
    return label


def relax(db,scope,child,score,depth,path,parent):
    # At equal length the literal path order is ordinary sorted-neighbor BFS order.
    return db.execute("""INSERT INTO walk_distances VALUES(?,?,?,?,?,?,1,NULL,1,-1)
        ON CONFLICT(scope,board) DO UPDATE SET depth=excluded.depth,path=excluded.path,
        parent=excluded.parent,dirty=1,changed_path=1
        WHERE excluded.depth<walk_distances.depth OR
        (excluded.depth=walk_distances.depth AND excluded.path<walk_distances.path)""",
        (scope,child,score,depth,path,parent)).rowcount


def next_activation(graph,key,label,level):
    if level<=0:return -1
    db=graph.db
    edge=db.execute('SELECT max(score) FROM walk_scored_edges WHERE board=? AND score<?',(label,level)).fetchone()[0]
    stored=stored_vector(graph,key)
    if stored is None:
        # Imported legacy closures may have no exact vector for unexamined pairs.
        return level-1
    floor=stored[1]//2
    lower=floor[floor<level]
    return max(edge if edge is not None else -1,int(lower.max()) if len(lower) else -1)


def certified(hit,pending):
    if hit is None:return False
    if pending is None or hit[1]==0:return True
    parent_depth=hit[1]-1
    return pending[0]>parent_depth or (pending[0]==parent_depth and pending[1]>=hit[2].rsplit('|',1)[0])


def run(graph,database,root,initial,floor,goal,max_centers,tie_limit,max_swaps,potentials,
        max_candidates,transport):
    db=graph.db
    schema(db)
    scope=hashlib.sha256(repr((root,floor,goal,max_swaps,bool(potentials),tie_limit)).encode()).hexdigest()
    level=min(initial['score'],goal-1)
    with db:
        db.execute('INSERT OR IGNORE INTO walk_widen VALUES(?,?,0)',(scope,level))
        relax(db,scope,root,initial['score'],0,root,None)
    work=graph.work
    work.update(edge_rows=0,band_expansions=0,path_propagations=0,activations=0,depth_closed=0,potential_closed=0)
    def goal_row():
        return db.execute("""SELECT d.board,d.depth,d.path FROM walk_distances d JOIN optima o ON o.key=d.board
                    WHERE d.scope=? AND d.score>=? ORDER BY d.depth,
                    substr(d.path,1,length(d.path)-length(d.board)-1),d.score DESC,
                    json_extract(o.result,'$.components'),d.path LIMIT 1""",(scope,goal)).fetchone()
    def get(key):
        return json.loads(db.execute('SELECT result FROM optima WHERE key=?',(key,)).fetchone()[0])
    while True:
        db.execute('BEGIN IMMEDIATE')
        level,done=db.execute('SELECT level,done FROM walk_widen WHERE scope=?',(scope,)).fetchone()
        row=db.execute("""SELECT board,depth,path,parent,expanded_floor,changed_path FROM walk_distances
                      WHERE scope=? AND dirty=1 ORDER BY depth,path LIMIT 1""",(scope,)).fetchone()
        hit=goal_row()
        reached=certified(hit,(row[1],row[2]) if row else None)
        if reached or done:
            db.commit()
            break
        if row is None:
            if level==floor:
                db.execute('UPDATE walk_widen SET done=1 WHERE scope=?',(scope,))
            else:
                # No other action can change this graph until the next eligible
                # star band or an already priced edge first enters the floor.
                wake=db.execute('SELECT max(next_floor) FROM walk_distances WHERE scope=?',(scope,)).fetchone()[0]
                level=max(floor,wake if wake is not None else -1)
                db.execute('UPDATE walk_widen SET level=? WHERE scope=?',(level,scope))
                work['activations']+=db.execute("""UPDATE walk_distances SET dirty=1 WHERE scope=?
                                                   AND next_floor>=?""",(scope,level)).rowcount
            db.commit()
            continue
        if work['centers']>=max_centers or time.monotonic()>=graph.deadline:
            db.commit()
            break
        key,depth,path,parent,old_floor,changed_path=row
        board=get(key)['codes']
        db.commit()
        skip=max_swaps is not None and depth>=max_swaps
        if skip:work['depth_closed']+=1
        elif max_swaps is not None and potentials:
            from r30_bounds import budget_upper, defect_packings
            bound=ExchangeBound(graph.model,board)
            skip=budget_upper(bound,defect_packings(bound),max_swaps-depth)<goal
            work['potential_closed']+=int(skip)
        whole=old_floor is None or changed_path
        if not skip:
            if not advance(graph,database,key,board,level,deadline=graph.deadline,
                           candidate_limit=max_candidates,parent=parent if transport else None):break
            with db:
                label=index_center(db,key)
            arc_sql="""SELECT v.board,e.score FROM walk_scored_edges e JOIN walk_vertex_labels v ON v.id=e.neighbor
                       WHERE e.board=? AND e.score>=?"""
            args=(label,level)
            if not whole:
                arc_sql+=' AND score<?'
                args+=(old_floor,)
            arcs=list(db.execute(arc_sql+' ORDER BY v.board',args))
            work['edge_rows']+=len(arcs)
            work['path_propagations' if whole else 'band_expansions']+=1
            # Recover archived score-only rows and resolve component ties outside
            # the short queue commit. A stopped visit can safely reread these edges.
            try:
                high=db.execute('SELECT max(score) FROM walk_distances WHERE scope=?',(scope,)).fetchone()[0]
                for child,score in arcs:
                    graph.check_time()
                    raw=db.execute('SELECT result FROM optima WHERE key=?',(child,)).fetchone()
                    if raw is None:
                        encoded=db.execute('SELECT seed FROM walk_legacy WHERE board=?',(child,)).fetchone()[0]
                        graph.optimum(child,json.loads(encoded),level,tie_limit if score>=high else 1)
                    elif tie_limit>1 and score>=high:
                        graph.optimum(child,json.loads(raw[0])['codes'],level,tie_limit)
                    high=max(high,score)
                    if db.in_transaction:db.commit()
            except ExpansionStopped:
                db.rollback()
                break
            wake=next_activation(graph,key,label,level)
        else:
            arcs=[]
            wake=-1
        db.execute('BEGIN IMMEDIATE')
        now=db.execute("""SELECT board,depth,path FROM walk_distances WHERE scope=? AND dirty=1
                          ORDER BY depth,path LIMIT 1""",(scope,)).fetchone()
        current=db.execute('SELECT level FROM walk_widen WHERE scope=?',(scope,)).fetchone()[0]
        if now!=row[:3] or current!=level:
            db.commit()
            continue
        for child,score in arcs:
            relax(db,scope,child,score,depth+1,path+'|'+child,key)
        db.execute("""UPDATE walk_distances SET dirty=0,expanded_floor=?,changed_path=0,next_floor=?
                      WHERE scope=? AND board=?""",(level,wake,scope,key))
        db.commit()
        work['centers']+=1
    # A budget interruption may already have a feasible target while a shorter
    # pending path still needs checking. Expose its witness and proof separately.
    db.execute('BEGIN IMMEDIATE')
    level,done=db.execute('SELECT level,done FROM walk_widen WHERE scope=?',(scope,)).fetchone()
    hit=goal_row()
    if hit:
        key,depth,path=hit
    else:
        key,depth,path=db.execute("""SELECT d.board,d.depth,d.path FROM walk_distances d JOIN optima o ON o.key=d.board
            WHERE d.scope=? ORDER BY d.score DESC,json_extract(o.result,'$.components'),d.depth,d.path LIMIT 1""",(scope,)).fetchone()
    remaining=db.execute('SELECT count(*) FROM walk_distances WHERE scope=? AND dirty=1',(scope,)).fetchone()[0]
    pending=db.execute('SELECT depth,path FROM walk_distances WHERE scope=? AND dirty=1 ORDER BY depth,path LIMIT 1',(scope,)).fetchone()
    reached=certified(hit,pending)
    count=db.execute('SELECT count(*) FROM walk_distances WHERE scope=?',(scope,)).fetchone()[0]
    best=get(key)
    db.commit()
    swaps=[]
    vertices=path.split('|')
    for frm,to in zip(vertices,vertices[1:]):
        prev,post=(s.split(':',1)[1].split(',') for s in (frm,to))
        changes=[i for i,(u,v) in enumerate(zip(prev,post)) if u!=v]
        if len(changes)!=2:raise RuntimeError('Stored path is not a physical transposition.')
        swaps.append(changes)
    return dict(work,objective='matched_edges',target=goal,score=best['score'],components=best['components'],
                codes=best['codes'],gold=graph.model.problem.validate(best['codes'])['valid'],floor=level,
                closed=count-remaining,open=remaining,swaps=swaps,reached=reached,feasible=bool(hit),
                shortest_at_floor=depth if reached else None,widest_floor=(initial['score'] if depth==0 else level)
                if reached else None,rotation_work=graph.oracle.work)
