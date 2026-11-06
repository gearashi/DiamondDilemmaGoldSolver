"""End-to-end first-240 stopping, independent loop checks, and persistent caching."""
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_edge_perfect_run(output,expected_components):
    status=read_json(output/'status.json')
    assert status['best_matched_edges']==240,status
    assert status['state']=='edge_perfect',status
    best=read_json(output/'best.json')
    edge=read_json(output/'edge-perfect.json')
    assert best['validation']['line_components']==expected_components,best
    assert best['validation']['matched_edges']==240
    assert best['validation']['all_closed']
    assert not best['validation']['valid']
    assert not best['solved']
    assert edge['codes']==best['codes']
    assert edge['validation']==best['validation']
    assert not (output/'solution.json').exists()
    assert not (output/'run.lock').exists()
    assert (output/'best.html').exists()
    assert (output/'checkpoint.npz').exists()
    with np.load(output/'checkpoint.npz',allow_pickle=False) as checkpoint:
        pending=checkpoint['pending']!=0
        assert np.any(pending),'The stopping candidate must remain pending in the checkpoint'
        boards=checkpoint['bestboards'][:,pending].T
        assert np.any(np.all(boards==np.asarray(best['codes']),axis=1))
    assert status['checkpoint']['phase']=='idle',status
    assert status['checkpoint']['completed_generation'] is not None,status
    metadata=read_json(output/'checkpoint-meta.json')
    assert metadata['generation']==status['checkpoint']['completed_generation']
    assert metadata['checkpoint_metrics']['compressed'] is False
    cache=status['duplicate_cache']
    assert isinstance(cache['gpu'],dict)
    assert cache['disk']['unique_entries']>=1,cache
    database=output/'positions.sqlite3'
    assert database.is_file()
    with database.open('rb') as stream:
        assert stream.read(16)==b'SQLite format 3\x00'
    return status,best


def run():
    with tempfile.TemporaryDirectory(prefix='diamond-runner-',dir=ROOT) as directory:
        td=Path(directory).resolve()
        assert td.parent==ROOT.resolve()
        tiles=[{'id':f'T{i}','number':i+1,'group':'silver',
                'segments':[[[s,5],[s,7]] for s in range(3)]} for i in range(160)]
        source=td/'fixture.json';source.write_text(json.dumps({'tiles':tiles}),encoding='utf-8')
        output=td/'run'
        cmd=[sys.executable,str(ROOT/'solve.py'),'--data',str(source),
             '--output',str(output),'--seconds','3','--replicas','128','--cache-slots','64']
        def execute(extra=()):
            return subprocess.run(cmd+list(extra),cwd=ROOT,capture_output=True,text=True,timeout=90)
        first=execute()
        assert first.returncode==0,(first.stdout,first.stderr)
        first_status,first_best=check_edge_perfect_run(output,240)
        first_config=read_json(output/'run-config.json')
        assert not first_config['resumed']
        assert first_config['cache_slots']==64
        assert first_config['stop_on_matched_edges']==240
        namespace=first_status['duplicate_cache']['disk']['namespace']

        second=execute(['--resume'])
        assert second.returncode==0,(second.stdout,second.stderr)
        second_status,second_best=check_edge_perfect_run(output,240)
        second_config=read_json(output/'run-config.json')
        assert second_config['resumed']
        assert second_best['revalidated_previous_record']
        assert second_status['duplicate_cache']['disk']['namespace']==namespace
        assert second_status['duplicate_cache']['disk']['session_hits']>=1,second_status
        assert second_best['codes']==first_best['codes']

        # Keep the same endpoint masks but join three gold loops through one tile.
        # The disk namespace must include segment connectivity, not just edge masks.
        tiles[0]['segments']=[[[0,5],[1,7]],[[1,5],[2,7]],[[2,5],[0,7]]]
        source.write_text(json.dumps({'tiles':tiles}),encoding='utf-8')
        third=execute(['--resume'])
        assert third.returncode==0,(third.stdout,third.stderr)
        third_status,_=check_edge_perfect_run(output,238)
        assert third_status['duplicate_cache']['disk']['namespace']!=namespace
        assert third_status['duplicate_cache']['disk']['session_misses']>=1,third_status

        # Increasing replica count seeds the new population from a preserved,
        # validated old checkpoint; it is not an exact RNG-state resume.
        old_checkpoint_hash=hashlib.sha256((output/'checkpoint.npz').read_bytes()).hexdigest()
        fourth=execute(['--resume','--replicas','256'])
        assert fourth.returncode==0,(fourth.stdout,fourth.stderr)
        fourth_status,_=check_edge_perfect_run(output,238)
        fourth_config=read_json(output/'run-config.json')
        assert fourth_config['seeded_from_checkpoint'],fourth_config
        assert not fourth_config['resumed'],fourth_config
        assert fourth_status['replicas']==256
        preserved=output/'runs'/fourth_config['run_id']/'starting-checkpoint.npz'
        assert preserved.exists(),preserved
        assert hashlib.sha256(preserved.read_bytes()).hexdigest()==old_checkpoint_hash
        with np.load(output/'checkpoint.npz',allow_pickle=False) as checkpoint:
            assert checkpoint['boards'].shape==(160,256)

        # A corrupted endpoint inventory must fail before any GPU search.
        tiles[0]['segments'][0][0][1]=4
        source.write_text(json.dumps({'tiles':tiles}),encoding='utf-8')
        invalid=execute()
        assert invalid.returncode!=0
        assert 'inventory cannot close' in invalid.stderr,invalid.stderr
        assert not (output/'run.lock').exists()
        return {
            'passed':True,'first_240_stops':True,'perfect_edge_multiloop_not_called_solved':True,
            'edge_perfect_certificate_saved':True,'winner_pending_in_checkpoint':True,
            'resume_stops_again':True,'persistent_cache_hit_after_resume':True,
            'cache_namespace_includes_connectivity':True,
            'replica_resize_warm_start_preserves_checkpoint':True,
            'invalid_inventory_rejected':True,'renderer_written':True,
        }


if __name__=='__main__':
    names=('test_runner.py','solve.py','gpu_engine.py','kernels.cu','geometry.py',
           'validator.py','position_cache.py','io_utils.py','render_board.py',
           'checkpoint_writer.py')
    def source_hashes():
        return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    started=time.monotonic();before=source_hashes()
    result=run()
    after=source_hashes()
    result.update(started_at=started_at,runtime_seconds=time.monotonic()-started,
                  command=list(sys.orig_argv),working_directory=str(ROOT),exit_status=0,
                  python=sys.version,platform=platform.platform(),
                  source_sha256_before=before,source_sha256_after=after,
                  execution_sources_unchanged=before==after)
    print(json.dumps(result,indent=2))
    (ROOT/'runner-tests.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
