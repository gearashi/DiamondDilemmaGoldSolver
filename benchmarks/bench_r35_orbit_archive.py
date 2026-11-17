"""Read-only canonical identities across the actual saved routes, fresh boards and plateau."""
from contextlib import closing
from pathlib import Path
import argparse,json,sqlite3
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r34_centers import archive_boards
from r35_center_orbits import CenterOrbits
from search_problem import SearchProblem


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--runtime',type=Path,default=Path('../runtime'));cli.add_argument('--output',type=Path)
    args=cli.parse_args();model=BoardModel(SearchProblem(read_json(args.tiles)));orbits=CenterOrbits(model)
    report={'proper_rotations':len(orbits.transforms),'sets':{}}
    def record(name,iterator):
        values=set(iterator);classes={orbits.canonical(p)[0] for p in values}
        report['sets'][name]=dict(physical=len(values),orbits=len(classes),rotated_duplicates=len(values)-len(classes))
        return values
    fresh=record('fresh_cpu_cuda',(tuple(c//3 for c in board) for board in archive_boards([args.runtime/'fresh-cpu.sqlite',args.runtime/'fresh-cuda.sqlite'])))
    centers=record('published_centers',(tuple(c['pieces']) for c in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']))
    parse=lambda s:tuple(map(int,s.split(':')[-1].split(',')))
    with closing(sqlite3.connect((args.runtime/'plateau-206.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        optimum=record('optima',(parse(k) for k, in db.execute('SELECT key FROM optima')))
        legacy=record('legacy_optimum',(parse(k) for k, in db.execute('SELECT DISTINCT perm FROM optimum')))
        bounds=record('plateau_bounds',(parse(k) for k, in db.execute('SELECT board FROM walk_bounds')))
        queue=record('plateau_queue',(parse(k) for k, in db.execute('SELECT DISTINCT board FROM walk_queue')))
        prior=record('legacy_centers',(parse(k) for k, in db.execute('SELECT DISTINCT perm FROM centers')))
    record('cross_archive_union',fresh|centers|optimum|legacy|bounds|queue|prior)
    from r34_centers import select
    from tempfile import TemporaryDirectory
    boards=list(archive_boards([args.runtime/'fresh-cpu.sqlite',args.runtime/'fresh-cuda.sqlite']))
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:
        results=[]
        for count in (8,len(boards),len(boards)):
            _,_,trial=select(model.problem,boards[:count],Path(tmp)/'selection.db',(2,2,3),208,3)
            results.append(dict(input=count,priced=trial['fresh_mask_prices'],reused=trial['reused_mask_prices'],
                                excluded=len(trial['cap_excluded'])))
        report['incremental_selection']=results
        from search_fixtures import planted_gold
        from r34_cycle_orders import CycleOrders
        from r31_routes import rank
        from r30_multicenter import run
        tiles,geo,gold=planted_gold(181,1,4);held=SearchProblem(tiles,geo);small=BoardModel(held)
        orbit=CenterOrbits(small);f=(2,4);move=CycleOrders(f)[0];center=list(gold)
        for i in range(6):center[i]=gold[move[i]]
        transform=next(i for i,t in enumerate(orbit.transforms) if set(t.cells[:6])!=set(range(6)))
        changed=sorted(orbit.transforms[transform].cells[:6]);point=rank(changed,small.free)
        images=[orbit.forward(center,i) for i in range(10)]
        _,_,selection=select(held,images,Path(tmp)/'held-caps.db',f,0,10)
        kwargs=dict(families=(f,),target=len(held.edges),page=1,filters=())
        saved=Path(tmp)/'held-witness.db';run(held,[center],saved,start=0,stop=1,**kwargs)
        direct=run(held,[images[transform]],Path(tmp)/'held-direct.db',start=point,stop=point+1,**kwargs)
        transfer=run(held,[images[transform]],saved,start=point,stop=point+1,orbit_scores=True,**kwargs)
        assert held.validate(transfer['best']['codes'])['valid'] and direct['best']['score']==transfer['best']['score']
        report['heldout_transport']=dict(input_orientations=len(images),selected_centers=selection['candidates'],
            source_rank=0,requested_rank=point,score=transfer['best']['score'],direct_native=direct['work']['exact_solved'],
            transported_native=transfer['work'].get('exact_solved',0),transported_scores=transfer['work']['rotated_witnesses'])

    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
