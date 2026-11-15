"""Measure bounded replay of already-validated GPU host arrays through real dispatch."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from bench_r34_lane_resize import checkpoint, resize
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import solve, split_cubes, unpack_native
from gpu_lane_resize_round34 import ParkedStateCache
from search_problem import SearchProblem
from test_gpu_domain_round25 import joint_hall_fixture, independent_solutions
from validator import validate_arrangement


def compare(problem,target,saved,width,limit,early):
    counts = dict(peak_cached_bytes=0,retained_frames=0,hits=0,hit_bytes=0,decodes=0,decoded_bytes=0)
    class Measured(ParkedStateCache):
        def remember(self,native,state):
            before = self.bytes
            super().remember(native,state)
            if self.bytes > before:
                counts['retained_frames'] += 1
            counts['peak_cached_bytes'] = max(counts['peak_cached_bytes'],self.bytes)
        def take(self,native,engine,unpack):
            before = counts['decodes']
            record = super().take(native,engine,unpack)
            if before == counts['decodes']:
                counts['hits'] += 1
                counts['hit_bytes'] += record['state'].nbytes
            return record
    def cold(native,view):
        result = unpack_native(native,view)
        counts['decodes'] += 1
        counts['decoded_bytes'] += result['state'].nbytes
        return result
    stop_calls = iter((False,True))
    start = perf_counter()
    with patch('gpu_lane_resize_round34.ParkedStateCache',Measured), patch('gpu_domain_runner.PARKED_CACHE_BYTES',limit), patch('gpu_domain_runner.unpack_native',side_effect=cold):
        result = solve(problem,target=target,resume=saved,replicas=width,seconds=180,
                       stop=(lambda:next(stop_calls,True)) if early else None)
    assert counts['peak_cached_bytes'] <= limit
    checkpoint_ = result['checkpoint']
    proof = {'status':result['status'],'nodes':result['nodes'],'completed_lanes':result['stats']['gpu_domain_completed'],
             'support_checks':result['stats']['support_checks'],'codes':result.get('codes'),
             'frontier_digest':checkpoint_['frontier']['digest'],'checkpoint_digest':checkpoint_['digest']}
    return proof,{**counts,'limit':limit,'seconds':round(perf_counter()-start,5)}


def run_case(problem,target,lanes,stored_width,width,limits,independent):
    split = split_cubes([WorkScope.for_problem(problem,target).root],lanes)
    with closing(DomainCUDA(problem,split,target=target)) as engine:
        saved = checkpoint(engine)
        lane_bytes = engine.rows * 4
    first,_ = resize(problem,target,saved,stored_width)
    records = []
    for early in (True,False):
        expected = None
        results = []
        for limit in limits:
            proof, counts = compare(problem,target,first,width,limit,early)
            if expected is None:
                expected = proof
            assert proof == expected
            if early:
                assert proof['status'] == 'stopped'
            elif independent:
                assert tuple(proof['codes']) in independent
            else:
                assert proof['status'] == 'infeasible' and proof['completed_lanes'] == len(split)
            if results:
                assert results[0]['decoded_bytes'] - counts['decoded_bytes'] == counts['hit_bytes']
            results.append(counts)
        records.append({'early_pause':early,'answer':expected,'budgets':results})
    return {'scope':problem.fingerprint,'source_lanes':len(split),'source_width':stored_width,
            'active_width':width,'lane_bytes':lane_bytes,'arms':records}


def original_residual(tiles):
    codes = json.loads((Path(__file__).parent/'docs/R29_INCUMBENT.json').read_bytes())['codes']
    free = {0,1,2,3}
    problem = SearchProblem(tiles,fixed={i:code for i,code in enumerate(codes) if i not in free})
    report = validate_arrangement(tiles,codes,problem.board)
    mismatches = report['unmatched_edges']
    assert report['matched_edges'] == 207
    witness = next(edge for edge in mismatches if edge['cell_a'] not in free and edge['cell_b'] not in free)
    assert witness['points_a'] != witness['reversed_points_b']
    return problem,witness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    small = joint_hall_fixture()
    original,mismatches = original_residual(json.loads(args.tiles.read_bytes()))
    results = []
    for target in ('edges','gold'):
        for width in (1,2):
            results.append(run_case(small,target,8,3-width,width,(0,4000,16000,65536),independent_solutions(small,target)))
    limits = [v * 1024**2 for v in (0,1,4,8,16,32,64)]
    results.append(run_case(original,'gold',48,4,8,limits,set()))
    wide_limits = [v * 1024**2 for v in (0,8,16,32,64,128,512)]
    results.append(run_case(original,'gold',128,16,8,wide_limits,set()))
    large_limits = [v * 1024**2 for v in (0,64,128,256,512)]
    results.append(run_case(original,'gold',256,8,16,large_limits,set()))
    args.output.write_text(json.dumps({'unavoidable_fixed_seam':mismatches,'cases':results},indent=2)+'\n',encoding='utf8')
