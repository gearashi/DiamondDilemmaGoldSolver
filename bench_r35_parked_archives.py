"""Compare verified archive reuse to the released native slicer on identical CUDA states."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import subprocess
import tracemalloc
from unittest.mock import patch

from bench_r34_lane_resize import all_solutions, checkpoint, resize
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import COUNTERS, split_cubes
from gpu_lane_resize_round34 import restore_slices
from search_problem import SearchProblem
from test_gpu_domain_round25 import joint_hall_fixture, independent_solutions

BASELINE = '9abc6fc0dc3f1f2dc8537b05ed77c6e566fe2112'


def prior_slicer():
    source = subprocess.check_output(['git','show',BASELINE+':gpu_lane_resize_round34.py'],
                                     cwd=Path(__file__).resolve().parent)
    globals_ = {}
    exec(compile(source, 'v1.15.0/gpu_lane_resize_round34.py', 'exec'), globals_)
    return globals_['restore_slices']


def measured(implementation, problem, target, saved, width):
    counts = dict(archive_decodes=0,parked_reencodes=0,parked_reencoded_raw_bytes=0)
    def track(engine, archives, replicas, owner_ids, names, pack, unpack):
        def compress(view,snapshot):
            counts['parked_reencodes'] += 1
            counts['parked_reencoded_raw_bytes'] += snapshot['state'].nbytes
            return pack(view,snapshot)
        def decompress(payload,view):
            counts['archive_decodes'] += 1
            return unpack(payload,view)
        return implementation(engine,archives,replicas,owner_ids,names,compress,decompress)
    with patch('gpu_lane_resize_round34.restore_slices',side_effect=track):
        tracemalloc.start()
        try:
            out, _ = resize(problem,target,saved,width)
            counts['peak_traced_host_bytes'] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    return out, counts


def case(problem,target,original):
    previous = prior_slicer()
    boxes = split_cubes([WorkScope.for_problem(problem,target).root],8)
    with closing(DomainCUDA(problem,boxes,target=target)) as engine:
        if original:
            engine.propagate_root(20480)
        else:
            engine.step(16)
        saved = checkpoint(engine)
        device = engine.device
    rows = []
    expected = None if original else independent_solutions(problem,target)
    for width in (2,2,3,3,1,1,4,4,8):
        before, old = measured(previous,problem,target,saved,width)
        after, new = measured(restore_slices,problem,target,saved,width)
        if after != before:
            raise AssertionError(('baseline mismatch',width,[k for k in set(after)|set(before) if after.get(k) != before.get(k)]))
        if after['frontier'] != saved['frontier']:
            raise AssertionError(('frontier changed',width))
        changed = [key for key in COUNTERS if before['active_stats'].get(key,0) != saved['active_stats'].get(key,0)]
        if changed:
            raise AssertionError(('cumulative counters changed',width,changed))
        if expected is not None:
            assert all_solutions(problem,target,after) == expected
        rows.append(dict(width=width,parked=len(after.get('native_queue',())),before=old,after=new))
        saved = after
    return dict(device=device,scope=problem.fingerprint,solutions=None if original else len(expected),stages=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = {'baseline':BASELINE,'finite':case(joint_hall_fixture(),'edges',False),
              'original':case(SearchProblem(json.loads(args.tiles.read_bytes())),'gold',True)}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
