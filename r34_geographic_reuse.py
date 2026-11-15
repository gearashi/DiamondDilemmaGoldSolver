"""Checked cross-query geographic evidence on identical physical assumptions."""
import json

from r34_phase_cache import checked_witness


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r34_geo_domains(context TEXT,key TEXT,domains TEXT NOT NULL,
               PRIMARY KEY(context,key))""")
    db.execute("""CREATE TABLE IF NOT EXISTS r34_geo_boards(puzzle TEXT,key TEXT,
               PRIMARY KEY(puzzle,key))""")


def related(db, problem, context, domains, positive, key_for_domains):
    for raw, in db.execute("""SELECT q.result FROM r34_geo_boards b JOIN r32_gold q ON b.key=q.key
                              WHERE b.puzzle=?""",(problem.fingerprint,)):
        row=positive(json.loads(raw))
        if row is not None:
            return dict(row,reuse_kind='validated board')
    for key,raw,status in db.execute("""SELECT d.key,d.domains,json_extract(q.result,'$.status')
            FROM r34_geo_domains d JOIN r32_gold q ON q.key=d.key WHERE d.context=?""",(context,)):
        broader=tuple(int(v,16) for v in json.loads(raw))
        if status!='INFEASIBLE' or key!=key_for_domains(broader):
            raise ValueError('Stored geographic exclusion has inconsistent domain or physical bindings.')
        if len(broader)==len(domains) and all(bits & ~saved==0 for bits,saved in zip(domains,broader)):
            return dict(status='INFEASIBLE',cp_calls=0,reuse_kind='ancestor domains')
    return None


def verify_board(problem, row, domains, pins, require, forbid, reference, spec):
    checked=checked_witness(problem,row['codes'],row['phases'],domains,pins,require,forbid)
    if checked is None:
        return None
    pairs=set(checked['pairs'])
    missing=set(reference)-pairs
    if not spec['lower']<=len(missing)<=spec['upper']:
        return None
    if spec['outside_losses'] is not None and len(set(spec['outside'])-pairs)!=spec['outside_losses']:
        return None
    if any(not (set(group)-pairs) for group in spec['groups']):
        return None
    return dict(row,**checked)
