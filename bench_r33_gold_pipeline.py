"""Time the released and current Gold loss CLIs over the same complete original-puzzle layer."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles', default='data/tiles.json')
    cli.add_argument('--baseline', required=True, type=Path, help='Unmodified v1.13.0 source directory')
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    tiles = str(Path(args.tiles).resolve())
    roots = {'current':Path(__file__).resolve().parent, 'released':args.baseline.resolve()}
    rows = []
    for name in ('released','current','current','released'):
        start = perf_counter()
        command = [sys.executable, '-B', 'r32_gold.py', 'loss', '--upper','15','--outside','5',
                   '--seconds','0','--page','8192','--tiles',tiles]
        payload = json.loads(subprocess.check_output(command,cwd=roots[name],text=True))
        rows.append(dict(version=name,seconds=round(perf_counter()-start,4),result=payload))
    outcomes = [(r['result']['status'],r['result']['fingerprint'],r['result']['completed_prefixes'],
                 r['result']['work']['indexed_tails'],r['result']['work']['joint_queries']) for r in rows]
    assert len(set(outcomes))==1 and outcomes[0][0]=='INFEASIBLE' and outcomes[0][2]==266916
    sources = {key:{n:sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                    for n in ('r32_gold.py','r32_gold_syndrome.py')} for key,root in roots.items()}
    args.output.write_text(json.dumps(dict(sources=sources,rows=rows),indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
