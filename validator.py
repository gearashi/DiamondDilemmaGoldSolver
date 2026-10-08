"""Independent exact checker: tile uniqueness, every seam, one closed gold loop.

A placement code is 3*tile_index+rotation. Rotation adds to the side index;
endpoint position p is unchanged. Matching sides reverse p to 12-p.
Bit zero means position one in masks. Only rotations are permitted.
Segments crossing in a drawing stay separate; only explicit endpoints connect.
A pass certifies the encoded data, not its transcription from source images.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from geometry import build_board


def normalize_tiles(data):
    raw = data.get('tiles') if isinstance(data,dict) else data
    if not isinstance(raw,(list,tuple)):
        raise ValueError('tile data must be a list or an object with a tiles list')
    result, ids = [], set()
    for index,tile in enumerate(raw):
        if not isinstance(tile,dict):
            raise ValueError(f'tile {index} must be an object')
        tile_id = tile.get('id',str(index))
        if not isinstance(tile_id,(int,str)) or isinstance(tile_id,bool):
            raise ValueError(f'tile {index} has an invalid ID')
        if tile_id in ids:
            raise ValueError(f'duplicate tile ID {tile_id!r}')
        ids.add(tile_id)
        raw_segments = tile.get('segments')
        if not isinstance(raw_segments,(list,tuple)) or not raw_segments:
            raise ValueError(f'tile {tile_id!r} must have gold segments')
        segments, endpoints = [], set()
        for segment in raw_segments:
            if not isinstance(segment,(list,tuple)) or len(segment) != 2:
                raise ValueError(f'tile {tile_id!r} has a malformed segment')
            points = []
            for point in segment:
                if not isinstance(point,(list,tuple)) or len(point) != 2:
                    raise ValueError(f'tile {tile_id!r} has a malformed endpoint')
                side,p = point
                if type(side) is not int or type(p) is not int or not 0<=side<3 or not 1<=p<=11:
                    raise ValueError(f'tile {tile_id!r}: side must be 0..2 and position 1..11')
                endpoint = (side,p)
                if endpoint in endpoints:
                    raise ValueError(f'tile {tile_id!r} repeats endpoint {endpoint}')
                endpoints.add(endpoint)
                points.append(endpoint)
            # Same-side U-shaped segments are present in the real puzzle.
            segments.append(tuple(points))
        result.append({'id':tile_id,'segments':tuple(segments)})
    return result


def reverse_mask(mask):
    """Reverse eleven bits; bit zero is endpoint position one."""
    return sum(((int(mask)>>p)&1)<<(10-p) for p in range(11))


def orientation_masks(data):
    """Return [3*tile+rotation][side] masks, with bit p-1 for position p."""
    masks = []
    for tile in normalize_tiles(data):
        base = [0,0,0]
        for segment in tile['segments']:
            for side,p in segment:
                base[side] |= 1<<(p-1)
        for rotation in range(3):
            masks.append([base[(side-rotation)%3] for side in range(3)])
    return masks


class _DisjointSet:
    def __init__(self,size):
        self.parents = list(range(size))
        self.sizes = [1]*size
    def find(self,x):
        while self.parents[x] != x:
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x
    def join(self,a,b):
        a,b = self.find(a),self.find(b)
        if a==b:
            return
        if self.sizes[a] < self.sizes[b]:
            a,b = b,a
        self.parents[b] = a
        self.sizes[a] += self.sizes[b]


def validate_arrangement(data, arrangement, board=None):
    """Return a detailed report. Seam coordinates are checked without bit masks.

    Connectivity joins segment identities across seams; line intersections
    in a picture are never interpreted as junctions.
    """
    board = board or build_board()
    report = {
        'valid':False,'errors':[],'placement_count':0,'tile_count':0,
        'unique_tiles':False,'matched_edges':0,'total_edges':len(board.edges),
        'unmatched_edges':[],'line_components':None,'component_lengths':[],
        'all_closed':False,'total_segments':0,
    }
    try:
        tiles = normalize_tiles(data)
    except ValueError as error:
        report['errors'].append(str(error))
        return report
    report['tile_count'] = len(tiles)
    if isinstance(arrangement,dict):
        arrangement = arrangement.get('arrangement',arrangement.get('codes'))
    if not isinstance(arrangement,(list,tuple)):
        if hasattr(arrangement,'tolist'):
            arrangement = arrangement.tolist()
        else:
            report['errors'].append('arrangement must be a list of integer codes')
            return report
    if not isinstance(arrangement,(list,tuple)):
        report['errors'].append('arrangement must be one-dimensional')
        return report
    report['placement_count'] = len(arrangement)
    if len(tiles) != len(board.cells):
        report['errors'].append(f'expected {len(board.cells)} tiles, received {len(tiles)}')
    if len(arrangement) != len(board.cells):
        report['errors'].append(f'expected {len(board.cells)} placements, received {len(arrangement)}')
    if report['errors']:
        return report
    if any(type(code) is not int or not 0<=code<3*len(tiles) for code in arrangement):
        report['errors'].append('placement codes must be integers in [0,3*tile_count)')
        return report
    usage = Counter(code//3 for code in arrangement)
    report['unique_tiles'] = len(usage)==len(tiles)
    if not report['unique_tiles']:
        report['errors'].append('each physical tile must be used exactly once')
        report['missing_tiles'] = [tiles[i]['id'] for i in range(len(tiles)) if i not in usage]
        report['repeated_tiles'] = [tiles[i]['id'] for i in sorted(usage) if usage[i]!=1]
    endpoint_segment,cell_points,segment_count = {},[],0
    for cell,code in enumerate(arrangement):
        tile_index,rotation = divmod(code,3)
        sides = [set(),set(),set()]
        for segment in tiles[tile_index]['segments']:
            for side,p in segment:
                rotated = (side+rotation)%3
                sides[rotated].add(p)
                endpoint_segment[(cell,rotated,p)] = segment_count
            segment_count += 1
        cell_points.append(sides)
    report['total_segments'] = segment_count
    components = _DisjointSet(segment_count)
    connected = [0]*segment_count
    for a,sa,b,sb in board.edges:
        left = cell_points[a][sa]
        right = {12-p for p in cell_points[b][sb]}
        if left==right:
            report['matched_edges'] += 1
        else:
            report['unmatched_edges'].append({
                'cell_a':a,'side_a':sa,'cell_b':b,'side_b':sb,
                'points_a':sorted(left),'reversed_points_b':sorted(right),
            })
        for p in left & right:
            x,y = endpoint_segment[(a,sa,p)],endpoint_segment[(b,sb,12-p)]
            components.join(x,y)
            connected[x] += 1
            connected[y] += 1
    sizes = Counter(components.find(s) for s in range(segment_count))
    report['line_components'] = len(sizes)
    report['component_lengths'] = sorted(sizes.values(),reverse=True)
    report['all_closed'] = all(degree==2 for degree in connected)
    if report['matched_edges'] != report['total_edges']:
        count = report['total_edges']-report['matched_edges']
        report['errors'].append(f'{count} board edges have unmatched gold endpoints')
    if not report['all_closed']:
        report['errors'].append('gold paths contain unconnected endpoints')
    if len(sizes)!=1:
        report['errors'].append(f'gold paths form {len(sizes)} components, expected one')
    report['valid'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tiles',type=Path)
    parser.add_argument('arrangement',type=Path)
    parser.add_argument('--subdivision',type=int,default=4)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    data = json.loads(args.tiles.read_text(encoding='utf-8-sig'))
    arrangement = json.loads(args.arrangement.read_text(encoding='utf-8-sig'))
    report = validate_arrangement(data,arrangement,build_board(args.subdivision))
    rendered = json.dumps(report,indent=2)
    if args.output:
        args.output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
    return 0 if report['valid'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
