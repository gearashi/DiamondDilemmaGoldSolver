"""Measure repeated grouped-drop choices after an interruption on physical reference subsets."""
from collections import Counter
from math import inf
from pathlib import Path
from random import Random
import argparse
import json

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_group_drops import GroupDropTimeout, reuse_drops
from search_problem import SearchProblem


def compare(tiles):
    docs=Path(__file__).parent/'docs'
    p=SearchProblem(read_json(tiles))
    ref,_,gs=reference_groups(p,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(p).required
    full=OutsideSyndromes(p,ref,gs,req)
    remove=list(full.outside)
    Random(35).shuffle(remove)
    patterns=read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']
    result=[]
    for n,i,cap in [(20,0,15),(20,1,17),(24,1,16),(24,6,16),(32,3,17),(32,4,16)]:
        q=OutsideSyndromes(p,ref-set(remove[:n]),full.groups,req)
        assert q._joint_images is None
        local=[q.outside.index(full.outside[j]) for j in patterns[i] if full.outside[j] in q.outside]
        def run(path=(),stop=inf):
            choices=0
            def checkpoint(current):
                nonlocal choices
                choices+=1
                if choices==stop:
                    raise GroupDropTimeout(current)
            try:
                return reuse_drops(q,local,cap,inf,Counter(),resume=path,checkpoint=checkpoint),choices,()
            except GroupDropTimeout as exc:
                return None,choices,exc.path
        expected,total,_=run()
        cut=total*3//5
        interrupted,prefix,path=run(stop=cut)
        assert interrupted is None
        resumed,remaining,_=run(path)
        if resumed!=expected:
            raise AssertionError((resumed,expected))
        result.append(dict(omitted=n,case=i,cap=cap,verdict=resumed,full_choices=total,
                           interrupted_choices=prefix,cold_choices_after_interruption=total,
                           resumed_choices=remaining,inner_path=path))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=compare(args.tiles)
    args.output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    print([(r['verdict'],r['full_choices'],r['resumed_choices']) for r in rows])
