"""Reproduce every original-mesh departure claim without using the published lookup."""
import argparse
import hashlib
import json
from pathlib import Path

from r35_geographic_packing_check import check_packing
from search_problem import SearchProblem
from verify_r35_contact_proof import RawPuzzle
from verify_r35_path_mates import VectorReplay


def read(path):
    return json.loads(path.read_bytes())


def bonds(data):
    result = tuple(sorted(tuple(sorted(pair)) for pair in data))
    if any(a >= b for a,b in result) or len(set(v for e in result for v in e)) != 2*len(result):
        raise ValueError("Invalid physical side matching")
    return result


def audit(tiles, docs):
    data = read(tiles)
    p = SearchProblem(data)
    raw = RawPuzzle(data)
    vector = VectorReplay(p)
    work = []
    for filename, kind in (
        ("R35_GEOGRAPHIC_RESIDUALS.json","geographic_cores"),
        ("R35_EDGE_CONTACTS.json","edge_perfect_cores"),
        ("R35_PATH_CIRCUITS.json","gold_path_circuits"),
        ("R35_CONTACT_RESIDUALS.json","geographic_cores"),
        ("R35_CONTACT_RESIDUALS.json","edge_perfect_cores"),
        ("R35_MIP_DONOR_SEWINGS.json","geographic_cores"),
    ):
        checked = []
        poses = 0
        packing_nodes = 0
        for claim in read(docs / filename)[kind]:
            if kind == "gold_path_circuits":
                path, mate, loop = bonds(claim["path"]), bonds([claim["mate"]])[0], bonds(claim["loop"])
                _, matched, misses = vector.count(path,[{mate}])
                if (not all(raw.fits(a,b) for a,b in (*path,*loop)) or mate not in loop or mate in path
                        or not raw.sealed(loop) or matched <= 0 or misses != [0]):
                    raise ValueError("Missing path-to-proper-loop implication")
                checked.append((path,mate,loop))
                poses += matched
            else:
                group = bonds(claim)
                if not all(raw.fits(a,b) for a,b in group):
                    raise ValueError("Nonreciprocal claim")
                if kind == "geographic_cores":
                    result = check_packing(p,group,max_steps=100000)
                    if result["feasible"] is not False:
                        raise ValueError("Unproved geometric obstruction")
                    packing_nodes += result["nodes"]
                else:
                    geometric, matched = raw.poses(group)
                    if not geometric or matched:
                        raise ValueError("Unproved incidental contact")
                    poses += geometric
                checked.append(group)
        canonical = json.dumps(sorted(checked),separators=(",",":")).encode()
        work.append(dict(source=filename,kind=kind,count=len(checked),
                         sha256=hashlib.sha256(canonical).hexdigest(),packing_nodes=packing_nodes,
                         geometric_poses_or_matched_path_poses=poses))
    return dict(fingerprint=p.fingerprint,physical_tiles=p.n,physical_segments=p.total_segments,catalogues=work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles",type=Path,required=True)
    parser.add_argument("--docs",type=Path,default=Path(__file__).parent/"docs")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.tiles,args.docs),indent=2) + "\n"
    if args.output:
        args.output.write_text(text,encoding="utf8")
    print(text)
