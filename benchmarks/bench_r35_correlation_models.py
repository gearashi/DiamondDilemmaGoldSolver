"""Compare CP's native matching inference with explicit exact co-occurrences on real and planted scopes."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from r27_structure_fixtures import planted
from r28_exchange import read_json
import r30_gold
from r35_mate_correlations import post_correlations
from search_problem import SearchProblem


def compare(p,opts):
    result={}
    for mode in ((False,True,'signed') if opts.get('pins') else (False,True)):
        out=r30_gold.search(p,mate_correlations=mode,**opts)
        work=out.get('mate_correlations',{})
        row={k:out[k] for k in ('status','cp_calls','branches')}
        if out['status']=='GOLD':assert p.validate(out['codes'])['valid']
        row.update(work);result['signed_only' if mode=='signed' else 'explicit' if mode else 'prior']=row
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',default='data/tiles.json');cli.add_argument('--output',type=Path)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles));reports=[]
    for seed in (35,71,103,157):
        q,ds=planted(1,3,seed)
        for label,channel,turn in (('sparse',False,None),('full_channel',True,None),('mixed_words',False,10)):
            opts=dict(domains=ds,geography=True,seconds=15,max_rounds=48,channel=channel,turn_length=turn)
            reports.append(dict(seed=seed,mode=label,channel=channel,**compare(q,opts)))
    witness=read_json('docs/R35_CONNECTED_TOURS.json')['contract']
    for n in (0,80,160):
        pins={i:v for i,v in enumerate(witness['phases']) if i<n}
        opts=dict(pins=pins,project_blanks=True,tour_mode='contract',seconds=22,max_rounds=1)
        reports.append(dict(original=True,pinned_phases=n,**compare(p,opts)))
    from random import Random
    raw=read_json('docs/R35_CONNECTED_TOURS.json')['raw']['phases']
    for n in (80,120):
        pins={t:raw[t] for t in Random(355).sample(range(p.n),n)}
        opts=dict(pins=pins,project_blanks=True,tour_mode='contract',seconds=22,max_rounds=1)
        reports.append(dict(original=True,signed_callback=n,**compare(p,opts)))
    if args.output:args.output.write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(reports))

if __name__=='__main__':main()
