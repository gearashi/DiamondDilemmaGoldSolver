"""Exercise a real CUDA exhaustion under an isolated released, dormant-only code change."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import r35_classic_sources as policy
import systematic_evidence as check

ROOT = Path(__file__).resolve().parent
PORTABLE_CHANGE = '12031ec'


def restore(module,source_dir,run,hashes,scope,jobs):
    folder = run/'systematic'
    native = folder/'checkpoint.npz'
    with patch.object(module,'__file__',str(source_dir/'systematic_evidence.py')), np.load(native,allow_pickle=False) as archive:
        return module.reopen_terminal(run,native,folder/'exhaustion.json',archive,hashes,scope,jobs)


def benchmark(run, independent_output):
    marker = check.read(run/'systematic/exhaustion.json')
    origin = run/'runs'/marker['run_id']
    scope = {key:marker[key] for key in ('input_sha256','frontier_sha256','target','gpu_policy')}
    source = marker['source_sha256']
    normalized = check.archived_sources(origin,source)
    active = policy.active_image(origin,normalized)
    assert active and policy.image_digest(active) in policy.REVIEWED
    status = check.read(run/'status.json')
    assert status['verified_coverage_origin'] and status['state'] == 'exhausted'
    with TemporaryDirectory(dir=run.parent) as temporary:
        folder = Path(temporary)
        copy = folder/'changed';copy.mkdir()
        for name in source:
            shutil.copyfile(origin/name,copy/name)
        released = importlib.util.spec_from_file_location('released_classic_checker',origin/'systematic_evidence.py')
        prior = importlib.util.module_from_spec(released)
        released.loader.exec_module(prior)
        # Exact worktree image from the completed R35 portable v1 transfer.
        (copy/'constraint_work.py').write_bytes(subprocess.check_output(
            ['git','show',PORTABLE_CHANGE+':constraint_work.py'],cwd=ROOT))
        def capture():
            with TemporaryDirectory(dir=folder) as sub:
                return check.capture(copy,Path(sub),scope['input_sha256'])
        live = capture()
        changed = [n for n in source.keys()|live.keys() if source.get(n)!=live.get(n)]
        assert changed == ['constraint_work.py']
        try:
            restore(prior,copy,run,live,scope,marker['total_jobs'])
        except ValueError as exc:
            former = str(exc)
        else:
            raise AssertionError('released verifier unexpectedly accepted changed bytes')
        answer = restore(check,copy,run,live,scope,marker['total_jobs'])
        # Corroborate the dormant edge by actually exhausting the same GPU input
        # under that changed tree. Preserve this receipt for repeat validations.
        finished = independent_output/'systematic/exhaustion.json'
        if not finished.exists():
            if (independent_output/'systematic/checkpoint.npz').exists():
                raise RuntimeError('Incomplete independent control exists')
            command = [sys.executable,'-B',str(copy/'systematic_search.py'),'--backend','cuda','--target','gold',
                       '--replicas','480','--seconds','0','--nodes','128','--fixed-nodes',
                       '--data',str(copy/'tiles.json'),'--output',str(independent_output)]
            with (run.parent/'independent-classic.log').open('w',encoding='utf8') as log:
                subprocess.run(command,cwd=copy,check=True,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,'CUPY_CACHE_DIR':str(run.parent/'kernel-cache')})
        independent = check.read(finished)
        assert independent['source_sha256'] == live
        fresh_status = check.read(independent_output/'status.json')
        assert fresh_status['state'] == 'exhausted' and fresh_status['verified_coverage_origin']
        assert fresh_status['counters'] == status['counters'] == answer['counters']
        assert fresh_status['exhausted_jobs'] == answer['exhausted_jobs'] == marker['total_jobs']
        # Mutate the current image and its claimed hash together. Every active
        # source must still reject, including both kernels and the validator.
        altered = []
        for name in sorted(active):
            file = copy/name
            original = file.read_bytes()
            trial = original + b'\n'
            file.write_bytes(trial)
            fake = {**live,name:hashlib.sha256(trial).hexdigest()}
            try:
                restore(check,copy,run,fake,scope,marker['total_jobs'])
            except ValueError:
                altered.append(name)
            else:
                raise AssertionError(('accepted active change',name))
            finally:
                file.write_bytes(original)
        # A helper newly reachable only below the disabled endgame stays diagnostic.
        path = copy/'r35_dormant_probe.py'
        path.write_text('VALUE = 35\n',encoding='utf8')
        with (copy/'constraint_work.py').open('ab') as f:
            f.write(b'\nimport r35_dormant_probe\n')
        expanded = capture()
        assert set(expanded)-set(live) == {path.name}
        added = restore(check,copy,run,expanded,scope,marker['total_jobs'])
        assert added['counters'] == answer['counters']
        target = check.archived_sources(copy,{k:v for k,v in expanded.items() if k!='tiles.json'})
        incomplete = {k:v for k,v in target.items() if k != 'constraint_work.py'}
        assert not policy.compatible(origin,normalized,copy,incomplete,scope['gpu_policy'])
        # An original diagnostic file remains mandatory even if it was dormant.
        controls = []
        for flag in ('cpu_endgame','cpu_probe','proof_probes','new_unknown_policy'):
            altered_policy = {**scope['gpu_policy'],flag:True}
            allowed=policy.compatible(origin,normalized,copy,target,altered_policy)
            assert not allowed
            controls.append(flag)
        current = None
        with TemporaryDirectory(dir=folder) as sub:
            h = check.capture(ROOT,Path(sub),scope['input_sha256'])
            try:
                restore(check,ROOT,run,h,scope,marker['total_jobs'])
            except ValueError as exc:
                current = str(exc)
            else:
                raise AssertionError('accepted the genuinely changed Round 35 Gold pruning code')
    return {'release_run_id':marker['run_id'],'device':status['gpu'],'jobs':marker['total_jobs'],
            'original_and_independent_counters':answer['counters'],'complete_boards':marker['candidates_checked'],
            'reactivated_gpu_launches':0,'host_only':answer['terminal_reopened_on_host'],
            'diagnostic_files':len(normalized),'active_files':len(active),'reviewed_class':policy.image_digest(active),
            'changed_files':changed,'old_verifier_result':former,'active_mutations_rejected':altered,
            'added_dormant_dependency_accepted':path.name,'different_active_policy_rejected':controls,
            'working_tree_active_gold_change_rejected':current}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released-run',type=Path,required=True)
    parser.add_argument('--independent-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = benchmark(args.released_run.resolve(),args.independent_output.resolve())
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
