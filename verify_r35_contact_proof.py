"""Independently replay the original Gold contact certificate from raw piece endpoints."""
import argparse
import hashlib
from collections import Counter, defaultdict
import json
from pathlib import Path

from geometry import build_board
from validator import normalize_tiles


def require(truth, message):
    if not truth:
        raise ValueError(message)


class RawPuzzle:
    def __init__(self, data):
        tiles = normalize_tiles(data)
        self.board = build_board()
        self.neighbors = self.board.neighbor_cells
        self.facing = self.board.neighbor_sides
        self.count = len(tiles)
        scope = {'tiles':tiles, 'edges':self.board.edges, 'fixed':[]}
        self.fingerprint = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        self.end = {}
        self.slots = [set() for _ in range(3 * self.count)]
        self.phase = []
        for i, tile in enumerate(tiles):
            for segment in tile['segments']:
                ends = [(3*i+s, k) for s, k in segment]
                for x, y in (ends, ends[::-1]):
                    require(x not in self.end, "duplicate endpoint")
                    self.end[x] = y
                    self.slots[x[0]].add(x[1])
            sign = 0
            for s in range(3):
                self.phase.append(sign)
                sign ^= len(self.slots[3*i+s]) % 2
            require(sign == 0, "odd tile boundary")
        self.slots = [tuple(sorted(x)) for x in self.slots]

    def reference(self, codes):
        require(sorted(code//3 for code in codes)==list(range(self.count)), "invalid reference")
        found = set()
        for a,s,b,t in self.board.edges:
            left, right = codes[a], codes[b]
            u, v = 3*(left//3)+(s-left%3)%3, 3*(right//3)+(t-right%3)%3
            if self.fits(u,v):
                found.add(tuple(sorted((u,v))))
        return found

    def fits(self, a, b):
        return a//3 != b//3 and self.slots[a] == tuple(12-x for x in reversed(self.slots[b]))

    def sealed(self, bonds):
        # Walk complete connected components of endpoints. Each endpoint has one
        # tile arc and can acquire just one seam link. A fully paired component
        # smaller than the puzzle cannot join the final single circuit.
        across = {}
        check = set()
        for a,b in bonds:
            require(self.fits(a,b), "incompatible named bond")
            for k in self.slots[a]:
                x,y = (a,k),(b,12-k)
                require(x not in across and y not in across, "double seam")
                across[x],across[y]=y,x
                check.update((x,y))
        while check:
            component=set()
            stack=[check.pop()]
            closed=True
            while stack:
                x=stack.pop()
                if x in component:
                    continue
                component.add(x)
                stack.append(self.end[x])
                if x in across:
                    stack.append(across[x])
                else:
                    closed=False
            if closed and len(component)<len(self.end):
                return True
            check.difference_update(component)
        return False

    def confirm_required(self, required):
        # Independently enumerate complete reciprocal matchings in the local
        # side-signature class of each target. Test all ways to omit that target.
        singles=set()
        possible={}
        for a in range(len(self.slots)):
            possible[a]={b for b in range(len(self.slots)) if self.fits(a,b)}
            if len(possible[a])==1:
                singles.add(tuple(sorted((a,next(iter(possible[a]))))))
        for a,b in sorted(required-singles):
            group={i for i,p in enumerate(self.slots) if p in (self.slots[a],self.slots[b])}
            def counterexample(left,chosen):
                if not left:
                    return not self.sealed(chosen | singles)
                u=min(left,key=lambda i:(len(possible[i]&left),i))
                for v in sorted(possible[u]&left):
                    pair=tuple(sorted((u,v)))
                    if pair!=(a,b) and counterexample(left-{u,v},chosen|{pair}):
                        return True
                return False
            require(not counterexample(group,set()), "unproved compulsory bond")

    def phase_conflict(self, bonds):
        # Each compatible sewn pair contributes twice to its signed reciprocal
        # class. Flip one tile at a time to form that class's affine parity row.
        def label(side, flipped):
            slots=self.slots[side]
            sign=self.phase[side]^flipped
            return min((slots,sign),(tuple(12-x for x in reversed(slots)),sign^(len(slots)%2)))
        parity=defaultdict(int)
        constant=1<<self.count
        for side in range(len(self.slots)):
            a,b=label(side,0),label(side,1)
            parity[a]^=constant
            if a!=b:
                parity[a]^=1<<(side//3)
                parity[b]^=1<<(side//3)
        rows=list(parity.values())
        for a,b in bonds:
            require(self.fits(a,b), "nonreciprocal phase equation")
            rhs=self.phase[a]^self.phase[b]^(len(self.slots[a])%2)
            rows.append((1<<(a//3))^(1<<(b//3))^(rhs*constant))
        pivots={}
        for row in rows:
            while row and row!=constant:
                pivot=(row&-row).bit_length()-1
                if pivot not in pivots:
                    pivots[pivot]=row
                    break
                row^=pivots[pivot]
            if row==constant:
                return True
        return False

    def poses(self, bonds):
        # Try every labelled root, then physically place each neighbour.
        # Derive incidental seams afterwards by walking the occupied cells.
        links=defaultdict(list)
        for a,b in bonds:
            for u,v in ((a,b),(b,a)):
                links[u//3].append((u%3,v//3,v%3))
        root=min(links)
        geometric=perfect=0
        for cell in range(self.count):
            for rotation in range(3):
                poses={root:(cell,rotation)}
                occupancy={cell:(root,rotation)}
                stack=[root]
                valid=True
                while stack and valid:
                    u=stack.pop()
                    c,r=poses[u]
                    for side,v,back in links[u]:
                        d=(side+r)%3
                        other=self.neighbors[c][d]
                        orient=(self.facing[c][d]-back)%3
                        if v in poses:
                            valid=poses[v]==(other,orient)
                        elif other<0 or other in occupancy:
                            valid=False
                        else:
                            poses[v]=(other,orient)
                            occupancy[other]=(v,orient)
                            stack.append(v)
                        if not valid:
                            break
                if not valid:
                    continue
                require(len(poses)==len(links), "disconnected pose rule")
                geometric+=1
                for c,(u,r) in occupancy.items():
                    for d,other in enumerate(self.neighbors[c]):
                        if other in occupancy and c<other:
                            v,t=occupancy[other]
                            k=self.facing[c][d]
                            if not self.fits(3*u+(d-r)%3,3*v+(k-t)%3):
                                valid=False
                perfect+=valid
        return geometric,perfect


def audit(data, codes, proof):
    puzzle=RawPuzzle(data)
    require(puzzle.fingerprint==proof.get('fingerprint'), 'certificate belongs to another input')
    reference=puzzle.reference(codes)
    required=set(map(tuple,proof['required']))
    require(len(reference)==207 and sorted(reference)==list(map(tuple,proof['reference'])), "reference mismatch")
    puzzle.confirm_required(required)
    weight=0
    incidence=Counter()
    checks=Counter()
    for row in proof['weighted_cores']:
        bonds=set(map(tuple,row['bonds']))
        kind=row['kind']
        w=row['weight']
        require(type(w) is int and w>0 and bonds<=reference|required and bonds-required, "invalid charged term")
        if kind=='physical':
            valid=puzzle.sealed(bonds|required) or puzzle.phase_conflict(bonds|required)
        elif kind=='required_xor':
            valid=puzzle.phase_conflict(bonds|required)
        elif kind in ('edge_perfect','geographic'):
            geometric,perfect=puzzle.poses(bonds)
            checks['labelled_roots']+=3*puzzle.count
            valid=geometric==0 if kind=='geographic' else geometric>0 and perfect==0
        else:
            raise ValueError("unknown term type")
        require(valid,f"unproved {kind} term {sorted(bonds)}")
        for bond in bonds-required:
            incidence[bond]+=w
        weight+=w
        checks[kind]+=1
    maximum=max(incidence.values())
    bound=(weight+maximum-1)//maximum
    require(weight==proof['weight_sum'] and maximum==proof['maximum'] and bound==proof['lower'], "integer certificate mismatch")
    return dict(fingerprint=puzzle.fingerprint,checked=dict(checks),named_reference=len(reference),compulsory=len(required),
                weight=weight,maximum_load=maximum,minimum_lost=bound,maximum_retained=len(reference)-bound)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    here=Path(__file__).resolve().parent/'docs'
    parser.add_argument('--reference',type=Path,default=here/'R29_INCUMBENT.json')
    parser.add_argument('--certificate',type=Path,default=here/'R35_CONTACT_LOSS_CERTIFICATE.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    def read(path):
        return json.loads(path.read_bytes())
    result=audit(read(args.tiles),read(args.reference)['codes'],read(args.certificate))
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
