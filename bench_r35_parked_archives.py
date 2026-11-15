"""Compare verified archive reuse to the released native slicer on identical CUDA states."""
import argparse
import base64
import hashlib
from contextlib import closing
import json
from pathlib import Path
import subprocess
import tracemalloc
import zlib

import numpy as np
from unittest.mock import patch

from bench_r34_lane_resize import all_solutions, checkpoint, resize
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import COUNTERS, pack_native, solve, split_cubes, unpack_native
from gpu_lane_resize_round34 import restore_slices
from search_problem import SearchProblem
from test_gpu_domain_round25 import joint_hall_fixture, independent_solutions

BASELINE = '9abc6fc0dc3f1f2dc8537b05ed77c6e566fe2112'


ALIGNED = 'd73e737b1629d83a2e68a2371b496b5716f6e6c1'


def prior_slicer(commit):
    source = subprocess.check_output(['git','show',commit+':gpu_lane_resize_round34.py'],
                                     cwd=Path(__file__).resolve().parent)
    globals_ = {}
    exec(compile(source, commit+'/gpu_lane_resize_round34.py', 'exec'), globals_)
    return globals_['restore_slices']


def measured(implementation, problem, target, saved, width):
    counts = dict(archive_decodes=0,parked_reencodes=0,parked_reencoded_raw_bytes=0)
    def track(engine, archives, replicas, owner_ids, names, pack, unpack, **kwargs):
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


def native_identity(saved):
    """Check owned roots and every byte of active and sleeping lane state independently."""
    columns = []
    for archive in (saved['native'], *saved.get('native_queue', ())):
        data = zlib.decompress(base64.b64decode(archive['compressed_state'],validate=True))
        assert hashlib.sha256(data).hexdigest() == archive['state_sha256']
        array = np.frombuffer(data,dtype=np.uint32).reshape(archive['shape'])
        columns.extend((root,hashlib.sha256(array[:,i].tobytes()).hexdigest())
                       for i,root in enumerate(archive['cubes']))
    metadata = {k:v for k,v in saved.items() if k not in ('native_queue','native','digest')}
    return metadata, saved['native'], columns


def case(problem,target,original):
    implementations = {'released':prior_slicer(BASELINE), 'aligned':prior_slicer(ALIGNED),
                       'deferred':restore_slices}
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
    inputs = dict.fromkeys(implementations,saved)
    for width in (2,2,3,3,1,1,4,4,8):
        outputs = {}
        stage = {'width':width}
        for name, implementation in implementations.items():
            result, counts = measured(implementation,problem,target,inputs[name],width)
            outputs[name] = result
            counts['parked_widths'] = [len(a['cubes']) for a in result.get('native_queue',())]
            stage[name] = counts
            if result['frontier'] != saved['frontier'] or any(
                    result['active_stats'].get(key,0) != saved['active_stats'].get(key,0) for key in COUNTERS):
                raise AssertionError(('coverage or cumulative work changed',width,name))
        wanted = native_identity(outputs['released'])
        for name in ('aligned','deferred'):
            if native_identity(outputs[name]) != wanted:
                raise AssertionError(('native owner, lane byte or metadata differs',width,name))
        if expected is not None:
            assert all_solutions(problem,target,outputs['deferred']) == expected
        rows.append(stage)
        inputs = outputs
    return dict(device=device,scope=problem.fingerprint,solutions=None if original else len(expected),stages=rows)


def dispatch_control(target, width):
    problem = joint_hall_fixture()
    boxes = split_cubes([WorkScope.for_problem(problem,target).root],8)
    with closing(DomainCUDA(problem,boxes,target=target)) as engine:
        saved = checkpoint(engine)
    # One sequence parks sub-batches below the requested width, the other
    # parks oversized archives. Both go through the solver's normal dispatch.
    archive_width = 1 if width == 2 else 2
    expected = independent_solutions(problem,target)
    result = {}
    for name, slicer in [('released',prior_slicer(BASELINE)),('deferred',restore_slices)]:
        allocations = []
        work = {'decoded_frames':0,'decoded_raw_bytes':0,'encoded_frames':0,'encoded_raw_bytes':0}
        def decoded(native,view):
            result = unpack_native(native,view)
            work['decoded_frames'] += 1
            work['decoded_raw_bytes'] += result['state'].nbytes
            return result
        def encoded(view,snapshot=None):
            work['encoded_frames'] += 1
            work['encoded_raw_bytes'] += 4*view.rows*view.lanes
            return pack_native(view,snapshot)
        def tracked(*args,**kwargs):
            engine = DomainCUDA(*args,**kwargs)
            allocations.append(engine.lanes)
            return engine
        def without_cache(*args,**kwargs):
            return slicer(*args)
        with patch('gpu_lane_resize_round34.restore_slices',side_effect=without_cache):
            first,_ = resize(problem,target,saved,archive_width)
            second,_ = resize(problem,target,first,width)
            with patch('gpu_domain_round24.DomainCUDA',side_effect=tracked), patch('gpu_domain_runner.pack_native',side_effect=encoded), patch('gpu_domain_runner.unpack_native',side_effect=decoded):
                answer = solve(problem,target=target,replicas=width,resume=second,seconds=120)
        if expected:
            assert answer['status'] in ('solved','edge_perfect') and tuple(answer['codes']) in expected
        else:
            assert answer['status'] == 'infeasible' and answer['complete'] and not answer['checkpoint']['frontier']['boxes']
        result[name] = {'codec_work':work,'widths_launched':allocations,'status':answer['status'],'nodes':answer['nodes'],
                        'support_checks':answer['stats']['support_checks'], 'completed':answer['stats']['gpu_domain_completed']}
    assert all(result['released'][k] == result['deferred'][k] for k in result['released'] if k != 'codec_work')
    return {'target':target,'width':width,'oracle_solutions':len(expected),**result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = {'baseline':BASELINE,'aligned_commit':ALIGNED,'finite':case(joint_hall_fixture(),'edges',False),
              'original':case(SearchProblem(json.loads(args.tiles.read_bytes())),'gold',True),
              'dispatch':[dispatch_control(t,w) for t in ('edges','gold') for w in (1,2)]}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
