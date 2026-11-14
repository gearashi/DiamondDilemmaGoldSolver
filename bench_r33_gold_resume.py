"""Compare exact Gold checkpoint replay with the v1.13.0 runner on identical validated pages."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
import types
from unittest.mock import patch

import r32_gold
from r32_gold_bonds import inventory
from r32_gold_loss import digest, read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def legacy_runner(path):
    release = '41f1e9c980fc441c3151ff21b5f73ae12ca3d9d7'
    raw = (Path(path).read_bytes() if path else
           subprocess.check_output(['git','show',release+':r32_gold.py'],cwd=Path(__file__).parent))
    raw = raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest() != '3b6b83ecdbf22599bdfa1b50a61f6179173f8fff5a9666f2dde2e2d31e098704':
        raise ValueError('Supply the unmodified v1.13.0 r32_gold.py as the baseline')
    module = types.ModuleType('r32_gold_v113')
    exec(compile(raw,'r32_gold_v113.py','exec'),module.__dict__)
    return module.run_syndrome


def measure(run, args, dbpath, upper, page):
    counts = Counter()
    statements = Counter()
    compile_group = OutsideSyndromes.compile
    connect = sqlite3.connect
    def traced(*args,**kw):
        db = connect(*args,**kw)
        def trace(statement):
            head = statement.lstrip().split()[0]
            statements[head] += 1
        def progress():
            counts['sqlite_vm_hundreds'] += 1
            return 0
        db.set_trace_callback(trace)
        db.set_progress_handler(progress,100)
        return db
    def compile_one(query, groups):
        counts['group_projections'] += 1
        return compile_group(query,groups)
    start = perf_counter()
    with patch('sqlite3.connect',traced), patch.object(OutsideSyndromes,'compile',compile_one):
        result = run(*args,upper,None,dbpath,0,page=page)
    elapsed = perf_counter()-start
    return dict(status=result['status'],complete=result['all_outside_layers'],covered=result['completed_prefixes'],
                new_pages=result['new_pages'],work=result['work'],selects=statements['SELECT'],
                write_transactions=statements['BEGIN'],sqlite_vm_hundreds=counts['sqlite_vm_hundreds'],
                group_projections=counts['group_projections'],seconds=round(elapsed,5))


def backup(source,target):
    with closing(sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--legacy-source',help='v1.13.0 r32_gold.py when the release commit is unavailable locally')
    cli.add_argument('--checkpoint',help='Completed legacy database, copied read-only. Default generates a budget-14 control.')
    cli.add_argument('--upper',type=int,default=14)
    cli.add_argument('--output',required=True)
    options = cli.parse_args()
    old = legacy_runner(options.legacy_source)
    p = SearchProblem(read_json(options.tiles))
    ref,_,groups = reference_groups(p,read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes'])
    args = (p,ref,groups)
    packed,required,_ = OutsideSyndromes.layout(ref,groups,inventory(p).required)
    scope = digest([3,p.fingerprint,sorted(ref),sorted(required),[sorted(c) for c in packed],options.upper])
    report = dict(fingerprint=p.fingerprint,loss_budget=options.upper,scope=scope,rows=[])
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        archived = root/'archived.sqlite'
        if options.checkpoint:
            backup(options.checkpoint,archived)
        else:
            result = old(*args,options.upper,None,archived,0,page=64)
            if result['status']!='INFEASIBLE':
                raise ValueError('Expected a complete negative reference budget')
        with closing(sqlite3.connect(archived)) as db:
            pages = db.execute('SELECT count(*) FROM r32_loss_pages WHERE scope=?',(scope,)).fetchone()[0]
            row = db.execute('SELECT errors,lo,hi FROM r32_loss_pages WHERE scope=? ORDER BY hi-lo DESC LIMIT 1',(scope,)).fetchone()
        if not row:
            raise ValueError('The supplied checkpoint contains no matching loss pages')
        layer,lo,hi = row
        for mode in ('original_pages','fragmented_ledger','partial_resume'):
            sample = root/(mode+'.sqlite')
            backup(archived,sample)
            with closing(sqlite3.connect(sample)) as db, db:
                if mode!='original_pages':
                    db.execute('DELETE FROM r32_loss_pages WHERE scope=? AND errors=? AND lo=?',(scope,layer,lo))
                    if mode=='fragmented_ledger':
                        # Split a genuinely proved region into 1024 narrow, adjacent legacy pages.
                        width = max(1,(hi-lo+1023)//1024)
                        restored = [(i,min(hi,i+width)) for i in range(lo,hi,width)]
                    else:
                        # Keep both ends but reopen 31 ranks inside a genuinely completed interval.
                        first = lo+(hi-lo)//2
                        last = min(hi,first+31)
                        restored = [(a,b) for a,b in ((lo,first),(last,hi)) if a<b]
                    db.executemany('INSERT INTO r32_loss_pages VALUES(?,?,?,?)',
                                   [(scope,layer,a,b) for a,b in restored])
                count = db.execute('SELECT count(*) FROM r32_loss_pages WHERE scope=?',(scope,)).fetchone()[0]
            paired = {}
            for label,run in (('v113',old),('current',r32_gold.run_syndrome)):
                control = root/(mode+label+'.sqlite')
                backup(sample,control)
                paired[label] = measure(run,args,control,options.upper,16)
            before,after = paired.values()
            for key in ('status','complete','covered','new_pages','work'):
                if before[key]!=after[key]:
                    raise AssertionError((mode,key,before[key],after[key]))
            if not after['complete'] or (mode!='partial_resume' and after['new_pages']):
                raise AssertionError('Replay scope did not match the supplied negative budget')
            report['rows'].append(dict(mode=mode,initial_pages=count,**paired))
        report['original_page_count'] = pages
    Path(options.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
