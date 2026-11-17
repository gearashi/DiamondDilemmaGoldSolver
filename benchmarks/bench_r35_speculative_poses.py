"""Replay the same phase-compatible speculative physical joins with both pose builders."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import Assemblies,physical_bonds
from r32_fresh import _poses,speculate
from r32_fresh_phases import fit
from search_problem import SearchProblem


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(35400,35416)))
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    model=BoardModel(p)
    a=Assemblies(model)
    rows=[]
    for seed in args.seeds:
        phases=fit(p,a.bonds,{0:0},seed=seed,effort=.2)
        if 'phases' not in phases:
            print(seed,'missing phases',flush=True)
            continue
        hint=fresh_start(model,np.random.default_rng(seed))
        for merges in (3,8,16):
            counts=Counter()
            class Neighbors:
                def __getitem__(self,c):
                    counts['baseline_edge_probes']+=1
                    return p.board.neighbors[c]
            observed=SimpleNamespace(n=model.n,legal=model.legal,problem=SimpleNamespace(board=SimpleNamespace(neighbors=Neighbors())))
            def trace(m,members,bonds):
                table=_poses(observed,members,bonds)
                counts['baseline_tables']+=1
                counts['baseline_rows_emitted']+=len(table)
                return table
            with patch('r32_fresh._poses',trace):
                old,plain=speculate(model,a,hint,phases['phases'],np.random.default_rng(seed),merges)
            new,info=speculate(model,a,hint,phases['phases'],np.random.default_rng(seed),merges,incremental=True)
            for board in (old,new):
                bonds=physical_bonds(model,model.legal_board(board))
                if not a.bonds<=bonds or a.forbidden&bonds:raise AssertionError('Bad physical mate.')
            assert info['joined_rows_emitted']==counts['baseline_rows_emitted']
            same=bool(np.array_equal(old,new))
            if not plain['backed_out'] and not same:raise AssertionError('Changed a deterministic unblocked seed.')
            rows.append(dict(seed=seed,merges=merges,old=plain,new=info,baseline=dict(counts),same_codes=same,
                             released_score=p.validate(old.tolist())['matched_edges'],incremental_score=p.validate(new.tolist())['matched_edges']))
            atomic_json(args.output,dict(input=p.fingerprint,records=rows))
            print(seed,merges,counts['baseline_edge_probes'],info['join_probes'],same,flush=True)


if __name__=='__main__':
    main()
