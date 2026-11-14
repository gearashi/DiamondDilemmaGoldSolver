"""Cross-board conflict reuse, disjoint loss branches, mating bounds and checkpoint replay."""
import argparse
import json
from pathlib import Path
import sqlite3
from contextlib import closing
from tempfile import TemporaryDirectory
from r31_geographic import physical_bonds, departure_cores
from r32_gold import run_syndrome
from r32_gold_bonds import inventory
from r32_gold_loss import (read_json,reference_groups,geographic,first_loss_partition,master,pose_lower,
                           dropout, Certificates)
from r32_gold_partition import maximize_disjoint
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--fresh',action='append',default=[],help='R30 independent-start database (read-only)')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    ref,cert,groups=reference_groups(problem,board)
    disjoint=maximize_disjoint(groups)
    choices=first_loss_partition(disjoint)[2]
    report=dict(branch_options=choices)
    rules=inventory(problem)
    with TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'follow.sqlite')
        branch_tests=[]
        for i in (0,len(choices)-1):
            keep,drop,_=first_loss_partition(disjoint,path=[i])
            response=geographic(problem,ref,disjoint,cert.rows,[i],16,16,1,2,path)
            branch_tests.append(dict(index=i,required=len(keep),forbidden=len(drop),
                                     status=response['status'],calls=response.get('cp_calls')))
        report['geographic']=branch_tests
        base=run_syndrome(problem,ref,groups,12,None,path,30,256)
        later=run_syndrome(problem,ref,groups,12,None,path,30,100)
        report['reopen']=[base,later]
    pool=Certificates(problem)
    pool.add(cert.clauses())
    others=[]
    for fn in args.fresh:
        with closing(sqlite3.connect(Path(fn).resolve().as_uri()+'?mode=ro',uri=True)) as db:
            row=db.execute('SELECT result FROM r30_starts ORDER BY score DESC,seed LIMIT 1').fetchone()
        if row:
            obj=json.loads(row[0])
            r=physical_bonds(problem,obj['codes'])
            found=departure_cores(problem,r,64,23,1)
            before=len(pool.rows)
            verified=Certificates(problem)
            verified.add(found['cores'])
            pool.add(verified.clauses())
            others.append(dict(codes=obj['codes'],matched=len(r),source_cores=len(verified.rows),
                               applicable_on_reference=sum(c<=ref for c in verified.rows),
                               union_before=before,union_after=len(pool.rows)))
    report['cross_board']=others
    report['joint_reference_counts']=dict(admitted=len(pool.rows),applicable=sum(r<=ref for r in pool.rows))
    # Seed a replay with a grouping learned on the reference. Both passes charge every native call.
    first=master(problem,ref,cert,12,2,rules.required,rules.forbidden)
    prev=len(pool.rows)
    pool.add(first['clauses'])
    second=master(problem,ref,pool,12,2,rules.required,rules.forbidden)
    report['separation_replay']=dict(first=first['calls'],second=second['calls'],
                                     first_status=first['status'],second_status=second['status'],
                                     clauses_before=prev,clauses_after=len(pool.rows))
    re=pose_lower(problem,ref,[c for c in pool.rows if c<=ref],board,2)
    report['pose_bound_after']=re
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
