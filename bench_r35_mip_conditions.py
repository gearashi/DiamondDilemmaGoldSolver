"""Compare signed feasibility and work on identical integer programs and loss caps."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from time import monotonic

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_contact_mip import build_model, decide
from r35_signed_conditions import SignedConditions
from r35_mip_conditions import Conditions
from verify_r35_contact_wide import read_rows


RAW_31 = 0xce77aac6e89509e8867b457e8c580a233a5e5f22
SUPPORTED_37 = 0xd237aba7e91101e88041413e8c589902021e5126


class ComparedConditions(Conditions):
    """Benchmark-only variants sharing the production feasibility and price replay."""
    def __init__(self,rules,engine,phase,losses,mode):
        super().__init__(rules,engine,phase,losses,startup=mode)
        if mode in ('small','families','price','paired8','paired12','paired16'):
            # Exact SAT price uses linked mates only for its narrow, proved-equal scopes.
            # Broad profile pricing may be a weaker relaxation, so leave losses independent.
            priced = rules.native_cost_pairs() if mode=='price' else None
            physical_width = int(mode[6:]) if mode.startswith('paired') else 0
            if physical_width:
                # This is a stronger physical relaxation. A bounded signed
                # phase price can be cheaper when its profile search times out.
                if rules._scope is None:
                    raise ValueError('Physical pair controls need a bound matching-price scope.')
                eligible = {side for sides, _, _, _ in rules.families if len(sides)<=physical_width for side in sides}
                ref,_,required,_ = rules._scope
                priced = {bond for bond in ref-required if bond[0] in eligible or bond[1] in eligible}
            classes = []
            modeled = set()
            for sides, edges, _, _ in rules.families:
                if mode=='small' and len(sides)>8:
                    continue
                link = priced is not None and len(sides) <= (physical_width or rules._class_prices().limit)
                classes.append((sides,edges,link))
                if link:
                    modeled.update(sides)
            represented=set()
            for sides, edges, link in classes:
                by_side={s:[] for s in sides}
                for a,b,rhs in edges:
                    pair=(a,b)
                    choice=1-losses[pair] if link and pair in losses else engine.BoolVar('')
                    if link and pair in rules._scope[2]:
                        engine.Add(choice==1)
                    x,y=phase[a//3],phase[b//3]
                    if rhs:
                        engine.Add(choice <= x+y)
                        engine.Add(choice <= 2-x-y)
                    else:
                        engine.Add(choice <= 1+x-y)
                        engine.Add(choice <= 1-x+y)
                    represented.add(pair)
                    by_side[a].append(choice)
                    by_side[b].append(choice)
                for options in by_side.values():
                    engine.Add(sum(options)==1)
                self.work['native_mate_families']+=1
                self.work['native_mate_edges']+=len(edges)
            if priced is not None:
                for pair in priced-represented:
                    engine.Add(losses[pair]==1)
                if any((a in modeled or b in modeled) and (a,b) not in represented for a,b in rules._scope[2]):
                    engine.Add(False)
                self.work['native_mate_price_links']=len(priced)
                if physical_width:
                    self.work['stronger_physical_pair_width']=physical_width



def factory(mode):
    if mode == 'whole':
        return lambda *args: None
    return lambda *args: ComparedConditions(*args,mode)


def run(tiles, output, modes, cases, seconds, scope, first_feasible, pattern=None):
    patterns = {'21':0xce55aaa6e89109fca073547e8c580ba32e0e5fa2,
                '22':0xd237aba7e99101e88041403e8c589902025e5326,
                '31':RAW_31,'37':SUPPORTED_37}
    def restricted(*args, **kwargs):
        engine,phase,losses=build_model(*args,**kwargs)
        if pattern:
            sample=patterns[pattern]
            for a,b in outside:
                x,y=phase[a//3],phase[b//3]
                engine.Add(x+y==1 if ((sample>>(a//3))^(sample>>(b//3)))&1 else x==y)
        return engine,phase,losses

    p,ref,initial,required=problem_scope(tiles)
    directory=Path(__file__).resolve().parent/'docs'
    physical=initial[:13]
    _,_,outside=OutsideSyndromes.layout(ref,physical,required)
    groups=list(physical)
    if scope!='physical':
        geo=read_rows(directory/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores')
        paths=read_rows(directory/'R35_EDGE_CONTACTS.json','edge_perfect_cores')
        groups.extend(geo)
        groups.extend(g for g in paths if scope in ('complete','circuits') or len(g)<=13)
    if scope == 'circuits':
        from r35_path_mates import verified_circuits
        document = json.loads((directory/'R35_PATH_CIRCUITS.json').read_bytes())
        proved = verified_circuits(p,document['gold_path_circuits'],ref | required)
        if len(proved) != document['nondominated']:
            raise ValueError('A path-circuit premise failed its independent geometric and proper-loop replay.')
        groups.extend(proved)
    result=[]
    for case in cases:
        cap, kind, hint = case.split('/')
        cap=int(cap)
        for mode in modes:
            rules=SignedConditions(p,euler='e' in kind,mates='m' in kind,mate_cost='p' in kind)
            rules=rules.for_loss(ref,groups,required,cap)
            suggested={'31':RAW_31,'37':SUPPORTED_37,'cold':None}[hint]
            started=monotonic()
            with patch('r35_contact_mip.build_model',restricted):
                out=decide(p,ref,groups,required,outside,cap,seconds,conditions=rules,
                           hint=suggested,condition_factory=factory(mode),first_feasible=first_feasible)
            result.append(dict(case=case,pattern=pattern,scope=scope,seconds_limit=seconds,mode=mode,first_feasible=first_feasible,
                               seconds=monotonic()-started,status=out['status'],cost=out.get('loss_cost'),
                               phase=out.get('phase'),work=out['work']))
            output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            print(case,mode,out['status'],out.get('loss_cost'),out['work'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--modes',nargs='+',choices=('whole','lazy','short','warm','rules','small','families','price','paired8','paired12','paired16'),
                     default=['whole','warm','rules','small','families','price'])
    cli.add_argument('--cases',nargs='+',default=['31/em/31','37/em/37'])
    cli.add_argument('--scope',choices=('physical','wide','complete','circuits'),default='wide')
    cli.add_argument('--seconds',type=float,default=100)
    cli.add_argument('--pattern',choices=('21','22','31','37'))
    cli.add_argument('--first-feasible',action='store_true')
    run(**vars(cli.parse_args()))
