"""Independently test geographic path implications with all root positions in parallel."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r35_induced_contacts import InducedContacts
from r35_loss_witness import raw_sides
from search_problem import SearchProblem


class VectorReplay:
    def __init__(self, problem):
        self.n = problem.n
        self.step = InducedContacts(problem).step
        self.slots = tuple(slots for slots,sign in raw_sides(problem.tiles))

    def count(self,premise,conclusions):
        graph=defaultdict(list)
        for a,b in premise:
            graph[a//3].append((a%3,b//3,b%3))
            graph[b//3].append((b%3,a//3,a%3))
        root=min(graph)
        active=np.ones(3*self.n,dtype=bool)
        pose={root:np.arange(3*self.n)}
        pending=[root]
        while pending:
            tile=pending.pop()
            for s,other,t in graph[tile]:
                dest=self.step[s,t][pose[tile]]
                active &= dest>=0
                if other in pose:
                    active &= dest==pose[other]
                else:
                    for existing in pose.values():
                        active &= dest//3!=existing//3
                    pose[other]=dest
                    pending.append(other)
        if len(pose)!=len(graph):
            raise ValueError("Expected a connected premise.")
        geo=int(active.sum())
        hits={}
        relevant=set().union(*conclusions)
        for left,right in combinations(sorted(pose),2):
            for s in range(3):
                for t in range(3):
                    a,b=3*left+s,3*right+t
                    contacted=None
                    if self.slots[a]!=tuple(12-k for k in reversed(self.slots[b])):
                        contacted=self.step[s,t][pose[left]]==pose[right]
                        active &= ~contacted
                    if (a,b) in relevant:
                        hits[a,b]=self.step[s,t][pose[left]]==pose[right] if contacted is None else contacted
        matched=int(active.sum())
        uncovered=[]
        for q in conclusions:
            avoid=active.copy()
            for bond in q:
                if bond in hits:avoid &= ~hits[bond]
            uncovered.append(int(avoid.sum()))
        return geo,matched,uncovered


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--screen",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    rows=read_json(args.screen)
    if rows["fingerprint"]!=p.fingerprint:raise ValueError("Different source puzzle.")
    test=VectorReplay(p)
    result=dict(fingerprint=p.fingerprint,sources={})
    for name,block in rows["sources"].items():
        cases=clauses=roots=0
        for row in block["cases"]:
            qs=[frozenset(map(tuple,q)) for q in row["alternatives"]]
            geo,matched,uncovered=test.count(row["bonds"],qs)
            if (geo,matched)!=(row["geometric"],row["compatible"]) or not matched or any(uncovered):
                raise ArithmeticError(f"Claim failed independent geometric replay: {name}, {row}")
            cases+=1;clauses+=len(qs);roots+=3*p.n
        result["sources"][name]=dict(cases=cases,implications=clauses,labelled_roots=roots)
    atomic_json(args.output,result)


if __name__=="__main__":
    main()
