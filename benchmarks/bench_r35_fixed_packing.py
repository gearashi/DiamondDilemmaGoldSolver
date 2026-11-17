"""Check compiled fixed-code swap packs against a Boolean physical-seam objective."""
from pathlib import Path
from unittest.mock import patch
import argparse

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import AssignmentPrices, simultaneous_swaps
from r35_touch_packing import augment
from r35_fixed_swap_cp import exact
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--effort',type=float,default=.2)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(35340,35352)))
    args=parser.parse_args()
    rows=[]
    problem=SearchProblem(read_json(args.tiles))
    model=BoardModel(problem)
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        board=fresh_start(model,rng)
        if seed%2:
            for _ in range(8):
                board=model.block(board,model.independent_set(board,rng),rng)
        for extent,allowed in [('board',model.free),('pool',rng.choice(model.free,30,replace=False))]:
            captured=[]
            def collect(m,values,pairs,codes,weights,initial,mode):
                result,indices,work=augment(m,values,pairs,codes,weights,initial,mode)
                captured.append((m,values,pairs,codes,indices))
                return result,indices,work
            with patch('r35_touch_packing.augment',collect):
                simultaneous_swaps(model,board,allowed,AssignmentPrices(model),exact=True,touch='complete')
            _,data=exact(*captured[0],args.effort)
            rows.append(dict(seed=seed,extent=extent,**data))
            atomic_json(args.output,dict(input=problem.fingerprint,effort=args.effort,records=rows))
            print(seed,extent,data,flush=True)


if __name__=='__main__':
    main()
