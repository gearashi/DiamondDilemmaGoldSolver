"""Persist a contiguous prefix of one expensive surviving order catalogue."""
from collections import Counter
from hashlib import sha256
import json


class OrderSliceExpired(Exception):
    """The leased support page stopped after its last complete order slice."""


class OrderFrontier:
    def __init__(self,db,scope,quantum):
        self.db=db;self.scope=scope;self.quantum=quantum
        self.active=False
        self.exists=bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='r35_order_cursor'").fetchone())

    def enable(self):
        if self.active:return
        self.db.execute("""CREATE TABLE IF NOT EXISTS r35_order_cursor(
            scope TEXT,rank INTEGER,owner TEXT,catalogue TEXT,done INTEGER,payload TEXT,checksum TEXT,
            PRIMARY KEY(scope,rank,owner,catalogue))""")
        self.active=True
        self.exists=True

    def catalogue(self,family,orders,complete):
        """Only the same order sequence may consume or extend a saved ordinal prefix."""
        if not self.quantum or len(orders)<max(512,2*self.quantum):return None
        if orders is complete:
            sequence='cycle-orders-v1'
        else:
            digest=sha256()
            for order in orders:digest.update(bytes(order))
            sequence=digest.hexdigest()
        self.enable()
        return json.dumps([1,family,len(orders),sequence],separators=(',',':'))

    @staticmethod
    def digest(key,done,payload):
        value=json.dumps([key,done,payload],separators=(',',':'))
        return sha256(value.encode()).hexdigest()

    def load(self,rank,owner,catalogue,size):
        key=(self.scope,rank,owner,catalogue)
        row=self.db.execute('SELECT done,payload,checksum FROM r35_order_cursor WHERE scope=? AND rank=? AND owner=? AND catalogue=?',key).fetchone()
        if row:
            done,raw,digest=row
            if type(done) is int and 0<done<=size and self.digest(key,done,raw)==digest:
                item=json.loads(raw)
                return done,Counter(item['work']),{(a,r,tuple(f)) for a,r,f in item['deps']},item['best']
        return 0,Counter(),set(),None

    def save(self,rank,owner,catalogue,done,work,dependencies,best,leases,claim):
        key=self.scope,rank,owner,catalogue
        raw=json.dumps(dict(work=work,deps=sorted(dependencies),best=best),separators=(',',':'))
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            leases.fence_uncovered(claim,'r30_pages',rank,rank+1)
            self.db.execute('INSERT OR REPLACE INTO r35_order_cursor VALUES(?,?,?,?,?,?,?)',
                            (*key,done,raw,self.digest(key,done,raw)))

    def promoted(self,lo,hi):
        if self.exists:
            self.db.execute('DELETE FROM r35_order_cursor WHERE scope=? AND rank>=? AND rank<?',(self.scope,lo,hi))

    def best(self):
        if not self.exists:return None
        best=None
        for rank,owner,cat,done,raw,digest in self.db.execute(
                'SELECT rank,owner,catalogue,done,payload,checksum FROM r35_order_cursor WHERE scope=?',(self.scope,)):
            if self.digest((self.scope,rank,owner,cat),done,raw)!=digest:continue
            row=json.loads(raw)['best']
            if row and (best is None or row['score']>best['score']):best=row
        return best
