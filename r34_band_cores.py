"""Persistent semantic regional SAT cores, replayed independently before reuse."""
import json

from constraint_work import digest


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r34_band_cores(
        scope TEXT, budget INTEGER, owner TEXT, terms TEXT, verified INTEGER,
        checksum TEXT NOT NULL, PRIMARY KEY(scope,budget,owner,terms))""")

    db.execute("""CREATE TABLE IF NOT EXISTS r34_band_core_deferrals(
        scope TEXT, owner TEXT, terms TEXT, PRIMARY KEY(scope,owner,terms))""")


def defer(db, guard, scope, owner, core):
    """A bounded verifier already spent a call on this hint for this exact query."""
    terms = json.dumps(sorted(core),separators=(',',':'))
    with db:
        db.execute('BEGIN IMMEDIATE')
        guard()
        db.execute('INSERT OR IGNORE INTO r34_band_core_deferrals VALUES (?,?,?)',
                   (scope,owner,terms))


def store(db, guard, scope, budget, owner, core, verified):
    terms = json.dumps(sorted(core),separators=(',',':'))
    row = (scope, budget, owner, terms, int(verified))
    with db:
        db.execute('BEGIN IMMEDIATE')
        guard()
        db.execute('INSERT OR REPLACE INTO r34_band_cores VALUES (?,?,?,?,?,?)',(*row,digest(row)))


def candidates(db, scope, budget, owner):
    rows = db.execute("""SELECT scope,budget,owner,terms,verified,checksum FROM r34_band_cores AS c
                         WHERE scope=? AND budget>=? AND (owner!=? OR verified=1)
                           AND (verified=1 OR NOT EXISTS (
                               SELECT 1 FROM r34_band_core_deferrals AS d
                               WHERE d.scope=c.scope AND d.owner=? AND d.terms=c.terms))
                         ORDER BY verified DESC,budget DESC""",(scope,budget,owner,owner))
    result = {}
    for *row,check in rows:
        if digest(row) != check:
            raise ValueError('Stored regional SAT core checksum differs.')
        core = frozenset(map(tuple,json.loads(row[3])))
        result.setdefault(core,bool(row[4]))
    return result


def assumptions(core, thresholds, budget):
    literals = []
    for group,sign,value in sorted(core):
        if sign==1:
            literals.append(thresholds[group][value-1])
        elif value<budget and value<len(thresholds[group]):
            literals.append(-thresholds[group][value])
    return literals
