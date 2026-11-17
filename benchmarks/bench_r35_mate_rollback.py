"""Compare whole-cycle projection and legacy cleanup on saved first physical collisions."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r31_fresh import Engine,physical_bonds
from r35_forbidden_mates import ForbiddenMates
from search_problem import SearchProblem


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--inputs',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    engine=Engine(problem,bonds=True)
    rules=ForbiddenMates(engine.model,engine.assemblies.forbidden)
    seen=set()
    rows=[]
    for f in args.inputs:
        for record in read_json(f)['records']:
            for example in record['examples']:
                old=np.array(example['before'])
                trial=np.array(example['after'])
                digest=hashlib.sha256(np.array([old,trial],dtype='<u2').tobytes()).hexdigest()
                if digest in seen or rules.violations(old).any() or not rules.violations(trial).any():
                    continue
                seen.add(digest)
                projected,work=rules.project(old,trial)
                baseline=engine.assemblies.clear_forbidden(trial.copy(),np.random.default_rng(record['seed']+example['step']))
                scores={}
                for label,board in (('old',old),('trial',trial),('projected',projected),('legacy',baseline)):
                    if board is None:continue
                    engine.model.legal_board(board)
                    if label!='trial' and (rules.pairs & physical_bonds(engine.model,board)
                                          or not engine.assemblies.bonds<=physical_bonds(engine.model,board)):
                        raise ValueError('Reconstructed board failed physical validation.')
                    scores[label]=int(problem.validate(board.tolist())['matched_edges'])
                rows.append(dict(seed=record['seed'],step=example['step'],stage=example['stage'],digest=digest,
                                 scores=scores,work=work))
    atomic_json(args.output,dict(input=problem.fingerprint,records=rows))
    print(len(rows),Counter(r['stage'] for r in rows))


if __name__=='__main__':
    main()
