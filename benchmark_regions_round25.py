"""Finite deterministic work counters for retained regional proposals."""
import argparse,json
from pathlib import Path
from test_constraint_regions_round25 import incremental_comparison,durable_comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=(14,15));parser.add_argument('--output')
    args=parser.parse_args();runners={14:incremental_comparison,15:durable_comparison}
    report={str(number):runners[number]() for number in runners if args.candidate is None or args.candidate==number}
    text=json.dumps(report,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    print(text,end='')
