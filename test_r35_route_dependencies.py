"""Verify family/rank-specific receipts, acyclic chains, and unreleased cyclic dependencies."""
import json
import sqlite3
import unittest
from r32_route_store import PageProofs,publish_page,publish_scores,schema
from r35_route_dependencies import Dependencies


def setup():
    db=sqlite3.connect(':memory:');schema(db)
    db.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
    return db


def publish(db,lo,hi,owners,edges,best=None,*,typed=None,families=()):
    db.execute('INSERT INTO r30_pages VALUES(?,?,?,?,?)',('src',lo,hi,'{}',json.dumps(best)))
    publish_page(db,'src',lo,hi,owners,edges)
    if typed is not None:typed.publish(db,'src',lo,owners,families)


class RouteDependencyTests(unittest.TestCase):
    def test_arbitrary_depth_cycle_and_arriving_dependency(self):
        with setup() as db:
            for i in range(120):
                publish(db,i,i+1,['a'],{'a':[('a',i+1)]})
            proof=PageProofs(db,208)
            self.assertEqual(proof.proven_end('src','a',0),0)
            publish(db,120,121,['a'],{})
            self.assertGreaterEqual(proof.proven_end('src','a',0),1)
            publish(db,130,131,['z'],{'z':[('z',131)]})
            publish(db,131,132,['z'],{'z':[('z',130)]})
            self.assertEqual(proof.proven_end('src','z',130),130)

    def test_same_rank_different_families_and_dependent_subranges(self):
        a,b=(2,4),(3,3)
        with setup() as db:
            typed=Dependencies()
            typed.add('later',a,3,'prior',51,b)
            typed.add('later',a,4,'prior',52,a)
            publish(db,0,10,('prior','later'),{'later':[('prior',51),('prior',52)]},typed=typed,families=(a,b))
            publish(db,50,60,('prior','later'),{},dict(score=210))
            publish_scores(db,'src',50,('prior','later'),(a,b),{(0,a):210,(0,b):203})
            proof=PageProofs(db,208)
            self.assertEqual(proof.proven_end('src','later',2,family=a),3)
            self.assertGreaterEqual(proof.proven_end('src','later',3,family=a),4)
            self.assertEqual(proof.proven_end('src','later',4,family=a),4)
            self.assertEqual(proof.proven_end('src','later',0,family=b),10)
            self.assertEqual(proof.proven_end('src','later',4),4)
            lower=PageProofs(db,200)
            self.assertEqual(lower.proven_end('src','later',3,family=a),3)

    def test_legacy_union_and_bounded_sparse_representation(self):
        a,b=(2,4),(3,3)
        with setup() as db:
            typed=Dependencies(limit=1);typed.add('x',a,1,'y',9,b);typed.add('x',a,2,'y',8,a)
            publish(db,0,5,('x','y'),{'x':[('y',8),('y',9)]},typed=typed,families=(a,b))
            proof=PageProofs(db,208)
            self.assertEqual(proof.proven_end('src','x',0,family=a),0)
            self.assertEqual(proof.proven_end('src','x',0,family=b),5)
            publish(db,5,6,('x','y'),{'x':[('y',8)]})
            self.assertEqual(proof.proven_end('src','x',5,family=b),5)
            publish(db,8,10,('x','y'),{})
            self.assertEqual(proof.proven_end('src','x',3,family=a),5)
            self.assertEqual(proof.proven_end('src','x',5,family=b),6)

if __name__=='__main__':unittest.main()
