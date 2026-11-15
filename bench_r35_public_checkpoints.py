"""Observe durable checkpoints and live updates partway through a maximum-width public GPU sweep."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bench_r35_parked_crash import child_interpreter, preserve_solution
from io_utils import atomic_json
from systematic_evidence import check_receipt


ROOT = Path(__file__).resolve().parent


def run(tiles, output, replicas, backend):
    exe, env = child_interpreter()
    samples = []
    with TemporaryDirectory(prefix='public-page-timing-', dir=ROOT/'audit/r35') as tmp:
        directory = Path(tmp)/'run'
        command = [exe, '-B', str(ROOT/'systematic_search.py'), '--data', str(tiles),
                   '--output', str(directory), '--backend', backend, '--target', 'gold',
                   '--replicas', str(replicas), '--seconds', '0', '--fixed-nodes', '--nodes', '512',
                   '--gpu-features', json.dumps({'dynamic_mrv': True, 'forced': True}),
                   '--checkpoint-seconds', '2']
        with (Path(tmp)/'worker.log').open('wb') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                limit = time.monotonic()+600
                while len(samples)<2 and time.monotonic()<limit and proc.poll() is None:
                    time.sleep(.2)
                    try:
                        status = json.loads((directory/'status.json').read_bytes())
                        live = json.loads((directory/'live.json').read_bytes())
                        if status['state']!='running' or not status.get('nodes_checked'):
                            continue
                        generation = status.get('checkpoint', {}).get('completed_generation')
                        if not generation or (samples and generation<=samples[-1]['generation']):
                            continue
                        checkpoint = directory/status['search_profile']/'checkpoint.npz'
                        with np.load(checkpoint, allow_pickle=False) as archive:
                            hint = int(archive['exact_next_page_lane'])
                            saved = int(archive['exact_generation'])
                        if saved!=generation or not 0<hint<replicas:
                            continue
                        samples.append(dict(generation=saved, next_page_lane=hint, elapsed=status['elapsed_seconds'],
                                            live_elapsed=live['elapsed_seconds'], nodes=status['nodes_checked']))
                    except (KeyError, FileNotFoundError, PermissionError, json.JSONDecodeError):
                        continue
                if len(samples)!=2:
                    raise RuntimeError('Timed out or exited without two distinct interrupted-sweep checkpoints.')
                stopping = time.monotonic()
                (directory/'stop.request').touch()
                if proc.wait(timeout=150)!=0:
                    raise RuntimeError('The isolated public Stop failed.')
                stop_seconds = time.monotonic()-stopping
            except Exception as exc:
                log.flush()
                raise RuntimeError((str(exc),(Path(tmp)/'worker.log').read_bytes()[-1800:])) from exc
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=30)
                preserve_solution(directory, tiles, output.with_name('cadence-verified-gold.json'))
        final = json.loads((directory/'status.json').read_bytes())
        if final['state']!='stopped' or not final['verified_coverage_origin']:
            raise AssertionError('Expected a stopped checkpoint with verified origin.')
        checkpoint = directory/final['search_profile']/'checkpoint.npz'
        with np.load(checkpoint, allow_pickle=False) as archive:
            check_receipt(directory,str(archive['exact_source_origin']),int(archive['exact_generation']),
                          checkpoint,str(archive['exact_source_binding']))
            served = int(np.count_nonzero(archive['counters'][0]))
            active = int(np.count_nonzero(archive['exact_job_ids']>=0))
        if not 0<served<replicas or samples[0]['next_page_lane']==samples[1]['next_page_lane']:
            raise AssertionError('The two saved partial sweeps did not advance different pages.')
        result = dict(gpu=final['gpu'],replicas=replicas,native_budget=512,checkpoint_seconds=2,
                      samples=samples,stop_through_portable_save_seconds=stop_seconds,lanes_with_trials=served,
                      active_at_stop=active,archive_verified=True)
    atomic_json(output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--backend',choices=('cuda','webgpu'),default='cuda')
    cli.add_argument('--replicas',type=int,default=131072)
    args=cli.parse_args()
    run(args.tiles.resolve(),args.output.resolve(),args.replicas,args.backend)
