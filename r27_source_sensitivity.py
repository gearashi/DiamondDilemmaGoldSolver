"""Read-only diagnostics for the uncertainty of the published gold drawings."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from validator import validate_arrangement, orientation_masks


def summary(data, codes):
    result=validate_arrangement(data,codes)
    return {key:result[key] for key in ("matched_edges","line_components","all_closed","valid")}


def modified_tiles(original, index, segments):
    tiles=list(original["tiles"])
    tiles[index]={"id":tiles[index]["id"],"segments":segments}
    return tiles


def audit(folder, incumbent, count=8):
    from audit_endpoints import GROUPS
    from extract_tiles import extract_tile
    data=json.loads((folder/"tiles.json").read_text(encoding="utf-8-sig"))
    endpoint=json.loads((folder/"endpoint_audit.json").read_text())
    for source,digest in data["source_sha256"].items():
        if hashlib.sha256((folder/source).read_bytes()).hexdigest()!=digest or endpoint["source_sha256"][source]!=digest:
            raise ValueError(f"Input diagram checksum differs for {source}")
    codes=incumbent["codes"]
    baseline=summary(data,codes)
    if not validate_arrangement(data,codes)["unique_tiles"]:
        raise ValueError("A complete physical permutation is required")
    ranked=sorted((tile["extraction"]["pairing_margin"],i) for i,tile in enumerate(data["tiles"])
                  if tile["extraction"]["pairing_margin"] is not None)
    variants=[]
    for margin,i in ranked[:count]:
        tile=data["tiles"][i]
        name,start,_,_=next(g for g in GROUPS if g[1]<=i+1<g[1]+g[2])
        j=i+1-start
        with Image.open(folder/f"tiles{name}.gif") as source:
            x,y=j%8*88,j//8*80
            image=source.convert("RGB").crop((x,y,x+82,y+71))
        evidence=endpoint["tiles"][i]
        if evidence["id"]!=tile["id"]:raise ValueError("Endpoint audit tile IDs differ")
        segments,fit,_=extract_tile(image,evidence["endpoints"],report_alternatives=1)
        if segments!=tile["segments"]:raise ValueError("Source extraction differs from prepared input")
        alternate=fit["alternatives"][0]["segments"]
        variant=modified_tiles(data,i,alternate)
        if orientation_masks(variant)!=orientation_masks(data):raise AssertionError("Pairing changed a seam mask")
        paired=summary(variant,codes)
        if paired["matched_edges"]!=baseline["matched_edges"]:raise AssertionError("Pairing changed fixed-board seam score")
        # Deliberate, labelled perturbation. Direction determined by pixel evidence
        # for the competing position, without using the incumbent to choose a shift.
        strengths={(s,p):v for s,p,v in evidence["endpoint_evidence"]}
        occupied={tuple(p) for arc in segments for p in arc}
        moves=[(strengths[s,q]-strengths[s,p],k,t,s,p,q)
               for k,arc in enumerate(segments) for t,(s,p) in enumerate(arc)
               for q in (p-1,p+1) if 1<=q<=11 and (s,q) not in occupied]
        gap,k,t,s,p,q=max(moves)
        shifted=[[list(port) for port in arc] for arc in segments]
        shifted[k][t]=[s,q]
        endpoint_score=summary(modified_tiles(data,i,shifted),codes)
        variants.append(dict(tile=tile["id"],pairing_margin=margin,alternate_loss=fit["alternatives"][0]["loss"],
                             alternative_pairing=alternate,pairing_result=paired,shift=[s,p,q],
                             contrast_loss=round(-gap,4),synthetic_endpoint_result=endpoint_score))
    return dict(baseline=baseline,threshold=endpoint["threshold"],stable_threshold=endpoint["threshold_stability_interval"],
                minimum_accepted_contrast=endpoint["minimum_accepted_contrast"],
                maximum_rejected_contrast=endpoint["maximum_rejected_contrast"],variants=variants)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("incumbent",type=Path)
    p.add_argument("--data",type=Path,default=Path("data"))
    args=p.parse_args()
    print(json.dumps(audit(args.data,json.loads(args.incumbent.read_text(encoding="utf-8-sig"))),indent=2))

