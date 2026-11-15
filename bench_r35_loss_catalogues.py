"""Compare cross-layer loss tables and the cost of transferring immutable pair XORs."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter
import json

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def compare(tiles):
    p=SearchProblem(read_json(tiles))
    ref,_,groups=reference_groups(p,read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes'])
    args=(p,ref,groups,inventory(p).required)
    old=OutsideSyndromes(*args,reuse_suffix_tables=False)
    new=OutsideSyndromes(*args)
    arms=[]
    for cap in range(12,19):
        for errors in range(2,cap-len(new.groups)+1):
            prev=old._build_suffix_table(cap,errors)
            current=new._build_suffix_table(cap,errors)
            assert (prev.order_key,prev.prices,prev.packed_tails,prev.buckets,prev.split_buckets)==(
                current.order_key,current.prices,current.packed_tails,current.buckets,current.split_buckets)
            arms.append(dict(cap=cap,errors=errors,rows=len(current.packed_tails)))
    catalogue=new._catalogues[2][0]
    order=list(catalogue)
    cols=new._packed_cols
    width=max(1,(max(cols).bit_length()+7)//8)
    blob=b''.join(v.to_bytes(width,'little') for v in catalogue.values())
    checksum=sha256(blob).digest()
    with TemporaryDirectory() as folder:
        path=Path(folder)/'catalogue.sqlite'
        with sqlite3.connect(path) as db:
            db.execute('create table c(v blob, checksum blob)')
            db.execute('insert into c values(?,?)',(blob,checksum))
        times={'calculate':[],'open_decode':[]}
        for _ in range(10):
            t=perf_counter()
            computed={pair:cols[pair[0]]^cols[pair[1]] for pair in combinations(range(len(cols)),2)}
            times['calculate'].append(perf_counter()-t)
            t=perf_counter()
            db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
            try:
                b,tag=db.execute('select v,checksum from c').fetchone()
            finally:
                db.close()
            if sha256(b).digest()!=tag:
                raise AssertionError('corrupt payload')
            restored={pair:int.from_bytes(b[j*width:(j+1)*width],'little') for j,pair in enumerate(combinations(range(len(cols)),2))}
            times['open_decode'].append(perf_counter()-t)
            assert computed==restored==catalogue
        size=path.stat().st_size
    return dict(reference=p.fingerprint,layers=arms,control=dict(old.index_work),retained=dict(new.index_work),
                cross_process_transfer=dict(source_pairs=len(order),fixed_width=width,payload_bytes=len(blob),
                                           sqlite_bytes=size,seconds=times))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.tiles)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['control'],result['retained'])
