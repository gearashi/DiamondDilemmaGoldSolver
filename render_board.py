"""Render an auditable numbered face layout as a standalone HTML file."""
from html import escape
import json
from pathlib import Path
from geometry import build_board

def render(path, tiles, codes, report):
    board = build_board()
    out = ['<svg viewBox="0 0 2080 980" xmlns="http://www.w3.org/2000/svg">']
    out.append('<rect width="2080" height="980" fill="#101720"/>')
    for face in range(10):
        ox = 20 + (face % 5)*412
        oy = 65 + (face//5)*465
        out.append(f'<text x="{ox+190}" y="{oy-22}" fill="#eef3ff" text-anchor="middle" font-size="24">{"Top" if face<5 else "Bottom"} {face%5}</text>')
        def xy(i,j):
            return (ox+190+(i-j)*46, oy+(i+j)*79.674)
        for cell in [c for c in board.cells if c.face == face]:
            i,j = cell.i,cell.j
            ij = ((i,j),(i+1,j),(i,j+1)) if not cell.inverted else ((i+1,j),(i+1,j+1),(i,j+1))
            vv = [xy(*q) for q in ij]
            code=int(codes[cell.index]); tile=tiles[code//3]; rot=code%3
            color={'silver':'#657280','red':'#742f43','blue':'#243e70'}[tile['group']]
            points=' '.join(f'{x:.2f},{y:.2f}' for x,y in vv)
            out.append(f'<polygon points="{points}" fill="{color}" stroke="#bac5d8" stroke-width=".65"/>')
            for segment in tile['segments']:
                pp=[]
                for side,p in segment:
                    side=(side+rot)%3
                    a,b=vv[side],vv[(side+1)%3]
                    pp.append((a[0]+(b[0]-a[0])*p/12,a[1]+(b[1]-a[1])*p/12))
                cx=sum(x for x,y in vv)/3;cy=sum(y for x,y in vv)/3
                # Inward stubs make same-side connectors visible.
                inner=[(x+(cx-x)*.08,y+(cy-y)*.08) for x,y in pp]
                q=[pp[0],inner[0],inner[1],pp[1]]
                out.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in q)+'" fill="none" stroke="#ffd75c" stroke-width="1.8"/>')
            cx=sum(x for x,y in vv)/3;cy=sum(y for x,y in vv)/3
            number=escape(str(tile.get('number',tile['id'])))
            out.append(f'<text x="{cx:.2f}" y="{cy+4:.2f}" fill="white" text-anchor="middle" font-size="11" stroke="#101720" stroke-width="2.8" paint-order="stroke">{number}/{rot}</text>')
        out.append(f'<text x="{ox+190}" y="{oy+355}" fill="#aabbce" text-anchor="middle" font-size="15">Face {face}: cells {face*16}–{face*16+15}</text>')
    out.append('</svg>')
    state='VERIFIED SINGLE LOOP' if report.get('valid') else 'SEARCH CANDIDATE — NOT A SOLUTION'
    summary=escape(json.dumps(report,indent=2))
    doc='<!doctype html><meta charset="utf-8"><title>Diamond Dilemma Gold</title><style>body{background:#101720;color:#edf3ff;font:16px system-ui;margin:24px}h1{font-size:25px}svg{width:100%;min-width:1050px}pre{white-space:pre-wrap;background:#1c2836;padding:20px}a{color:#ffd75c}.board{overflow:auto}p{max-width:1000px;line-height:1.6}</style>'
    doc+=f'<h1>Diamond Dilemma Gold · {state}</h1><p>Labels are original source tile number / clockwise rotation (0, 1, or 2). Each rotation is 120°. The gold paths are drawn from the extracted data. The ten faces are shown separately; their seams are joined in the stored geometry.</p><div class="board">{"".join(out)}</div><p>Top face k has vertices (top apex, equator k, equator k+1). Bottom face k has vertices (bottom apex, equator k+1, equator k), indices modulo 5. Face corners in this drawing are ordered top, bottom-right, bottom-left.</p><pre>{summary}</pre>'
    Path(path).write_text(doc,encoding='utf-8')
