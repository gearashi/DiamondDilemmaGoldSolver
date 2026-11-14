"""Reuse wholly negative ranked pages across compatible center sets and thresholds."""
import json


def register(db,scope,binding,families,target,owners):
    db.execute("""CREATE TABLE IF NOT EXISTS r31_members(
        scope TEXT, domain TEXT, family TEXT, target INTEGER, owner TEXT,
        PRIMARY KEY(scope,family,owner))""")
    db.execute("""CREATE INDEX IF NOT EXISTS r31_negative_lookup ON
        r31_members(domain,family,owner,target)""")
    db.executemany("INSERT OR IGNORE INTO r31_members VALUES(?,?,?,?,?)",
                   ((scope,binding,json.dumps(f),target,owner) for f in families for owner in owners))
    db.commit()


def inherited_end(db,scope,binding,families,target,owners,start,stop):
    """An owner can be deferred to another page in a multicenter run.

    Only a singleton source has no deferred cross-center endpoints. Within
    that scope the earlier-radius certificates are complete independent proofs.
    A higher target can also consume a completed lower-target page if the
    recorded maximum among its threshold hits is lower than the new target.
    """
    until = stop
    for f in families:
        for owner in owners:
            records = db.execute("""
                SELECT p.hi,p.best FROM r31_members m JOIN r30_pages p ON p.scope=m.scope
                WHERE m.domain=? AND m.family=? AND m.owner=? AND m.target<=? AND m.scope!=?
                AND p.lo<=? AND p.hi>?
                AND NOT EXISTS(SELECT 1 FROM r31_members n WHERE n.scope=m.scope AND n.owner!=m.owner)
                """,(binding,json.dumps(f),owner,target,scope,start,start))
            end = max((hi for hi,best in records if best=='null' or json.loads(best)['score']<target),
                      default=start)
            until = min(until,end)
            if until==start:
                return start
    return until
