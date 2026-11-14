"""Reconstruct the published white solutions from their raster geometry and number disks."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from validator import validate_arrangement

SOURCE = "https://www.jaapsch.net/puzzles/images/diamdil/"
DIAGRAMS = {
    "solsilver1": ("silver", 1, 32, "d76ac732b93e9c80654b26f7e722840a0ec5d66a0f0652e6d6ce3855a64d1b5d"),
    "solsilver2": ("silver", 1, 32, "7caa86f4b1919f3fd3675719de8f32046941beb32647b36b145532ef09486fa6"),
    "solred": ("red", 33, 48, "a68dda4257ab116854922c6eda22ae486ef6e1ecb536c7880137e6fe225f901e"),
    "solblue": ("blue", 81, 80, "c04b39485df26cce6597facda26a097c53a2fc9b61bc798be757ff91c78931cd"),
}
SHEET_VERTICES = np.array([[40.5, 1.], [80.5, 70.25], [.5, 70.25]])
Y, X = np.mgrid[:71, :82]
PIXELS = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=1)
BARY = PIXELS @ np.linalg.inv(np.column_stack([SHEET_VERTICES, np.ones(3)]))
PURPLE = np.array([164, 74, 164])
PORTS = tuple(itertools.product(range(3), range(1,12)))


def sample(array, xy):
    x = np.clip(xy[..., 0], 0, array.shape[1]-1)
    y = np.clip(xy[..., 1], 0, array.shape[0]-1)
    ix, iy = x.astype(int), y.astype(int)
    jx, jy = np.minimum(ix+1, array.shape[1]-1), np.minimum(iy+1, array.shape[0]-1)
    fx, fy = x-ix, y-iy
    return ((1-fy)*((1-fx)*array[iy,ix]+fx*array[iy,jx])
            + fy*((1-fx)*array[jy,ix]+fx*array[jy,jx]))


def board_triangles(pix):
    vertices = []
    for row in range(round((pix.shape[0]-1)/69.282)):
        for col in range((pix.shape[1]-1)//40):
            # Lattice coordinates keep side identity exact across the unfolded net.
            if (row+col)%2 == 0:
                tri = ((col+1,row+1),(col,row),(col+2,row))
            else:
                tri = ((col+1,row),(col+2,row+1),(col,row+1))
            xy = screen(tri)
            if (xy[:,0].max()>pix.shape[1]+1 or
                    np.abs(pix[tuple(xy.mean(axis=0).round().astype(int)[::-1])].astype(int)-PURPLE).sum()<30):
                continue
            vertices.append(tri)
    return vertices


def screen(triangle):
    return np.asarray(triangle)*[40.,69.2820323]+[.5,.25]


def match_tiles(sheet, pix, cells, count):
    # The templates come from the piece sheet, independent of the solved board.
    # The numbered disks identify each physical tile and its rotation.
    selected = (((X-40.5)**2+(Y-47)**2<11**2)).ravel()
    bary = BARY[selected]
    templates=[]
    for i in range(count):
        x,y=i%8*88,i//8*80
        templates.append(sheet[y:y+71,x:x+82].min(axis=2).ravel()[selected])
    templates=np.array(templates,dtype=float)
    templates-=templates.mean(axis=1,keepdims=True)
    templates/=np.linalg.norm(templates,axis=1,keepdims=True)
    white=pix.min(axis=2).astype(float)
    matches=[]
    for cell in cells:
        xy=screen(cell)
        shifts=np.array(list(itertools.product(np.arange(-2,2.1,.5),repeat=2)))
        spots=np.array([bary@np.roll(xy,-r,axis=0) for r in range(3)])
        scans=sample(white,spots[:,None,:,:]+shifts[None,:,None,:])
        scans-=scans.mean(axis=2,keepdims=True)
        scans/=np.linalg.norm(scans,axis=2,keepdims=True).clip(1e-9)
        correlations=(scans@templates.T).max(axis=1)
        matches.append(correlations.T)
    matches=np.array(matches)
    weights=matches.max(axis=2)
    from r27_incumbent import assignment
    ids=assignment(np.rint(weights*1e6),np.isfinite(weights))
    codes=[3*int(i)+int(np.argmax(matches[c,i])) for c,i in enumerate(ids)]
    margins=[float(weights[c,i]-np.max(np.delete(weights[c],i))) for c,i in enumerate(ids)]
    return codes,margins


def edges_of(cells, name):
    sides, edges = {}, []
    for i,tri in enumerate(cells):
        for s in range(3):
            a,b = tri[s],tri[(s+1)%3]
            if (b,a) in sides:
                j,t = sides.pop((b,a))
                edges.append((j,t,i,s))
            else:
                sides[a,b] = (i,s)
    if name == "solblue":
        # Close the cut from the common apex (8,4) to the rim in the five-face fan.
        # Locate the two boundary rays of the missing 60-degree lattice sector.
        for d in range(4):
            left = ((8-2*d,4),(6-2*d,4))
            lower = ((8-d,4+d),(7-d,5+d))
            def take(edge):
                return sides.pop(edge) if edge in sides else sides.pop(edge[::-1])
            j,t = take(left)
            i,s = take(lower)
            edges.append((j,t,i,s))
    return edges,list(sides.values())


def fit_white_routes(sheet,diagram):
    from extract_tiles import DIST, GRID, CANDIDATES, ENDPOINTS
    nodes=np.asarray(CANDIDATES)
    # Whole-route evidence on two independent renderings. Ignore the number disk.
    valid=(BARY.min(axis=1)>.009)&((GRID[:,0]-40.5)**2+(GRID[:,1]-47)**2>14**2)
    is_gold=np.minimum(sheet[:,:,0].astype(float),sheet[:,:,1])-sheet[:,:,2]>18
    silver=np.median(sheet[:,:,0])<200 and np.median(sheet[:,:,2])<200
    background,contrast=(171,80) if silver else (4,248)
    def normalize(value):
        return np.clip((value.astype(float)-background)/contrast,0,1)
    val=normalize(sheet.min(axis=2)).ravel()
    other=normalize(diagram.reshape(sheet.shape[:2])).ravel()
    # Distance from known path to bright pixels as weighted line-ink likelihood.
    evidence=np.maximum(val*(~is_gold.ravel()),other)
    observed=evidence>.60
    core=(DIST<.85) & valid
    cover=(DIST<2.45) & valid
    bad=(core & (evidence<.23)).sum(axis=1)
    residual=observed & valid
    used=set();chosen=[]
    for step in range(7):
        gains=(cover&residual).sum(axis=1)-bad*4-6
        for i,(a,b) in enumerate(CANDIDATES):
            if a in used or b in used:gains[i]=-999
        best=gains.argmax()
        if gains[best]<=0:break
        chosen.append(best); used.update(CANDIDATES[best]);residual&=~cover[best]
    # Coordinate descent refits each complete path with all other paths fixed.
    # A discrete symmetric-difference objective charges missed ink and invented ink.
    for _ in range(5):
        changed=False
        for slot,old in enumerate(chosen):
            others=chosen[:slot]+chosen[slot+1:]
            occupied={k for idx in others for k in CANDIDATES[idx]}
            residual=observed & valid
            if others:residual &= ~np.any(cover[others],axis=0)
            quality=(cover&residual).sum(axis=1) - bad*4 - 6
            for k,(a,b) in enumerate(CANDIDATES):
                if a in occupied or b in occupied:quality[k]=-10000
            best=int(np.argmax(quality))
            if quality[best]>quality[old]:
                chosen[slot]=best;changed=True
        if not changed:break
    return [[ENDPOINTS[a],ENDPOINTS[b]] for a,b in nodes[chosen]]
# Small same-side turns near corners and paths obscured by the number disk
# can fit a neighboring grid position. These ports were read on the numbered
# white-only diagram, cross-checked against the separate physical tile sheet.
REVIEWED = {
    39: (((1,6),(1,11)), ((0,3),(2,8)), ((0,7),(1,1))),
    159: (((0,5),(2,6)), ((0,7),(1,4)), ((1,6),(1,8))),
    91: (((1,11),(2,11)), ((0,11),(1,3))),
    93: (((0,10),(1,7)), ((1,8),(2,1)), ((0,1),(2,10))),
    132: (((0,7),(1,4)), ((1,9),(2,2)), ((0,11),(1,1))),
}


def decode(data, name):
    group,first,count,digest=DIAGRAMS[name]
    path=data/(name+".gif")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError(f"Unexpected source bytes for {name}")
    with Image.open(path) as image:
        pix=np.asarray(image.convert("RGB"))
    from prepare_data import load_manifest, verify_source
    filename=f"tiles{group}.gif"
    source=next(row for row in load_manifest()["sources"] if row["name"]==filename)
    verify_source(data/filename,source)
    with Image.open(data/filename) as image:
        sheet=np.asarray(image.convert("RGB"))
    cells=board_triangles(pix)
    codes,margins=match_tiles(sheet,pix,cells,count)
    if len(cells)!=count or len({c//3 for c in codes})!=count:
        raise ValueError(f"{name}: wrong number of distinct raster IDs {len(cells)} {codes}")
    edges,open_sides=edges_of(cells,name)
    tiles=[None]*count
    for code,tri in zip(codes,cells):
        i,rotation=divmod(code,3)
        x,y=i%8*88,i//8*80
        white_only=sample(pix.min(axis=2),BARY@np.roll(screen(tri),-rotation,axis=0))
        paths=fit_white_routes(sheet[y:y+71,x:x+82],white_only)
        number=first+i
        if number in REVIEWED:
            paths=REVIEWED[number]
        tiles[i]={"id":number,"segments":paths}
    board=SimpleNamespace(cells=cells,edges=edges)
    report=validate_arrangement(tiles,codes,board)
    return tiles,codes,board,open_sides,margins,report


def embed_patch(source, target, faces):
    """Map the oriented net to selected physical faces, including its rim."""
    adjacency=[[None]*3 for _ in source.cells]
    for a,s,b,t in source.edges:
        adjacency[a][s]=(b,t)
        adjacency[b][t]=(a,s)
    allowed={c.index for c in target.cells if c.face in faces}
    for first in sorted(allowed):
        for rotation in range(3):
            assignment={0:(first,rotation)}
            used={first}
            stack=[0]
            while stack:
                cell=stack.pop()
                target_cell,r=assignment[cell]
                for side,connection in enumerate(adjacency[cell]):
                    other,side_back=target.neighbors[target_cell][(r+side)%3]
                    if connection is None:
                        if other in allowed:break
                        continue
                    neighbor,back=connection
                    actual=(other,(side_back-back)%3)
                    if other not in allowed:break
                    if neighbor in assignment:
                        if assignment[neighbor]!=actual:break
                    else:
                        if other in used:break
                        assignment[neighbor]=actual
                        used.add(other)
                        stack.append(neighbor)
                else:continue
                break
            else:
                if used==allowed:return [assignment[cell] for cell in range(len(source.cells))]
    raise ValueError("Unfolded patch does not embed in the requested faces")


def run_controls(data):
    """Compare the published orientation patterns to shuffled, inventory-equal controls."""
    from geometry import build_board
    target=build_board()
    embeddings={}
    results={}
    gold=json.loads((data/"tiles.json").read_text(encoding="utf-8-sig"))
    for name,(group,first,count,_) in DIAGRAMS.items():
        tiles,codes,board,open_sides,margins,report=decode(data,name)
        if not report["valid"]:raise ValueError(f"{name} failed the single-loop control: {report['errors']}")
        rotated=codes.copy()
        rotated[0]=3*(codes[0]//3)+(codes[0]+1)%3
        rotated_valid=validate_arrangement(tiles,rotated,board)["valid"]
        reversed_tiles=tiles.copy()
        cells_to_reverse=[cell for cell,code in enumerate(codes) if any(p!=6 for arc in tiles[code//3]["segments"] for _,p in arc)]
        t=codes[cells_to_reverse[0]]//3
        reversed_tiles[t]={"id":tiles[t]["id"],"segments":[[(s,12-p) for s,p in arc] for arc in tiles[t]["segments"]]}
        mirror_valid=validate_arrangement(reversed_tiles,codes,board)["valid"]
        if rotated_valid or mirror_valid:raise AssertionError(f"{name} failed the corrupted-orientation control")
        tests=[]
        rotations=np.array(codes)%3
        subset=gold["tiles"][first-1:first-1+count]
        for seed in range(32):
            rng=np.random.default_rng(seed)
            permutation=rng.permutation(count)*3
            pair=[validate_arrangement(subset,(permutation+r).tolist(),board)["matched_edges"]
                  for r in (rotations,rng.permutation(rotations))]
            tests.append(pair)
        faces={"silver":(5,6),"red":(7,8,9),"blue":tuple(range(5))}[group]
        embedding=embed_patch(board,target,faces)
        embeddings[name]=[(cell,(code+rotation)%3) for code,(cell,rotation) in zip(codes,embedding)]
        results[name]={"cells":count,"matched_edges":report["matched_edges"],"segments":report["total_segments"],
                       "line_components":report["line_components"],"open_sides":len(open_sides),
                       "rotated_valid":rotated_valid,"reflected_valid":mirror_valid,
                       "gold_pattern_and_shuffle_mean":np.mean(tests,axis=0).tolist(),"paired_controls":tests,
                       "sha256":DIAGRAMS[name][3]}
    full={}
    for silver in ("solsilver1","solsilver2"):
        pattern=np.zeros(160,dtype=int)
        for cell,rotation in embeddings[silver]+embeddings["solred"]+embeddings["solblue"]:
            pattern[cell]=rotation
        paired=[]
        for seed in range(32):
            rng=np.random.default_rng(seed)
            perm=rng.permutation(160)*3
            paired.append([validate_arrangement(gold,(perm+r).tolist(),target)["matched_edges"]
                           for r in (pattern,rng.permutation(pattern))])
        full[silver]={"pattern_and_shuffle_mean":np.mean(paired,axis=0).tolist(),"pairs":paired}
    results["whole_gold_board"]=full
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--fetch",action="store_true",help="Fetch the four checksum-verified white controls")
    args=parser.parse_args()
    if args.fetch:
        import urllib.request
        for name,(_,_,_,digest) in DIAGRAMS.items():
            path=args.data/(name+".gif")
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
            with urllib.request.urlopen(SOURCE+path.name,timeout=30) as response:payload=response.read(1024*1024)
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError(f"Unexpected source digest for {name}")
            path.write_bytes(payload)
    print(json.dumps(run_controls(args.data),indent=2))


if __name__=="__main__":
    main()
