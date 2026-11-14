"""Append-only sequence numbers for scoped, incremental coverage replay."""
import json


def install(db):
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='coverage_sequence'").fetchone():
        return
    with db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='coverage_sequence'").fetchone():
            return
        db.execute("""CREATE TABLE coverage_scopes(id INTEGER PRIMARY KEY, puzzle TEXT, target TEXT, original TEXT,
                   UNIQUE(puzzle,target,original))""")
        db.execute("""CREATE TABLE coverage_sequence(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   identity TEXT UNIQUE, scope_id INTEGER NOT NULL)""")
        db.execute("CREATE INDEX coverage_scoped_sequence ON coverage_sequence(scope_id,seq)")
        db.execute("CREATE TABLE coverage_epoch(id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
        db.execute("INSERT INTO coverage_epoch VALUES(1,0)")
        def fields(source):
            return [f"json_extract({source}.scope,'$.{key}')" for key in ('fingerprint','target','original')]
        def join(source):
            return " AND ".join(f"{col}={expr}" for col,expr in zip(('puzzle','target','original'),fields(source)))
        db.execute(f"""INSERT INTO coverage_scopes(puzzle,target,original)
                    SELECT DISTINCT {','.join(fields('c'))} FROM closed c""")
        db.execute(f"""INSERT INTO coverage_sequence(identity,scope_id) SELECT c.identity,s.id
                    FROM closed c JOIN coverage_scopes s ON {join('c')} ORDER BY c.rowid""")
        add = f"""
            INSERT INTO coverage_scopes(puzzle,target,original) VALUES({','.join(fields('NEW'))})
                ON CONFLICT(puzzle,target,original) DO NOTHING;
            INSERT INTO coverage_sequence(identity,scope_id)
                SELECT NEW.identity,id FROM coverage_scopes WHERE {join('NEW')};
        """
        db.execute(f"""CREATE TRIGGER coverage_insert AFTER INSERT ON closed BEGIN
            UPDATE coverage_epoch SET value=value+1 WHERE EXISTS
                (SELECT 1 FROM coverage_sequence WHERE identity=NEW.identity);
            DELETE FROM coverage_sequence WHERE identity=NEW.identity;
            {add} END""")
        db.execute(f"""CREATE TRIGGER coverage_update AFTER UPDATE ON closed BEGIN
            UPDATE coverage_epoch SET value=value+1;
            DELETE FROM coverage_sequence WHERE identity IN (OLD.identity,NEW.identity);
            {add} END""")
        db.execute("""CREATE TRIGGER coverage_delete AFTER DELETE ON closed BEGIN
            UPDATE coverage_epoch SET value=value+1;
            DELETE FROM coverage_sequence WHERE identity=OLD.identity;
            END""")


def scope_id(db, target):
    row = db.execute("""SELECT id FROM coverage_scopes WHERE puzzle=? AND target=? AND original=?""",
                     (target.fingerprint,target.target,json.dumps([hex(x) for x in target.original],separators=(',',':')))).fetchone()
    return row[0] if row else 0


def version(db, group):
    return db.execute("""SELECT value,coalesce((SELECT max(seq) FROM coverage_sequence WHERE scope_id=?),0)
                         FROM coverage_epoch WHERE id=1""", (group,)).fetchone()


def entries(db, group, after, through):
    return db.execute("""SELECT c.identity,c.scope,c.box,c.evidence,c.checksum
        FROM coverage_sequence s JOIN closed c ON c.identity=s.identity
        WHERE s.scope_id=? AND s.seq>? AND s.seq<=? ORDER BY s.seq""", (group,after,through))
