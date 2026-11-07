"""Independently validated semantic SAT witnesses with indexed eligibility.

These are complete EDGE witnesses for the current optimistic Gold relaxation,
not claimed Gold solutions. Every disk load and transported board is rescored
from raw tile endpoints. New verified Gold cuts invalidate only matching rows.
"""
from collections import OrderedDict,defaultdict
import hashlib,json
from pathlib import Path
from validator import validate_arrangement
from constraint_knowledge import revalidate_witness

STAT_KEYS=('witness_semantic_validations','witness_index_queries','witness_index_terms',
 'witness_index_hits','witness_cut_checks','witness_cut_invalidations','witness_useful_supports',
 'witness_evictions','witness_loaded','witness_symmetry_checks','witness_transported')


def _binding(problem):
    return (problem.fingerprint,tuple(tuple(sorted(set(row))) for row in problem.domains),
            json.dumps(problem.tiles,sort_keys=True,separators=(',',':')),tuple(problem.board.edges))


class SemanticWitnessPool:
    def __init__(self,problem,*,limit=128):
        if type(limit)is not int or not 1<=limit<=4096:raise ValueError('Invalid semantic witness limit.')
        self.problem=problem;self.binding=_binding(problem);self.limit=limit
        self.rows=OrderedDict();self.index=defaultdict(set);self.serial=0;self.cuts=set();self.blocked=set()
        self.stats=dict.fromkeys(STAT_KEYS,0)

    def _check(self):
        if _binding(self.problem)!=self.binding:raise ValueError('Semantic witness scope changed.')

    def _validate(self,codes):
        self._check()
        if (not isinstance(codes,(tuple,list)) or len(codes)!=self.problem.n
                or any(type(v)is not int or v not in row for v,row in zip(codes,self.binding[1]))):
            raise ValueError('Witness violates original domains.')
        report=validate_arrangement(self.problem.tiles,list(codes),self.problem.board)
        self.stats['witness_semantic_validations']+=1
        if not (report['unique_tiles'] and report['matched_edges']==report['total_edges'] and report['all_closed']):
            raise ValueError('Semantic witness fails independent edge validation.')
        return tuple(codes)

    def add(self,codes,check=lambda:None):
        check();row=self._validate(codes);check();self._check()
        if row in self.rows:return False
        self.serial+=1;self.rows[row]=[self.serial,0]
        for c,v in enumerate(row):self.index[c,v].add(row)
        if any(all(row[c]==v for c,v in cut) for cut in self.cuts):self.blocked.add(row)
        while len(self.rows)>self.limit:
            def utility(candidate):
                unique=sum(len(self.index[c,v])==1 for c,v in enumerate(candidate))
                created,uses=self.rows[candidate]
                return (candidate not in self.blocked,unique,uses,created)
            victim=min(self.rows,key=utility);self.rows.pop(victim);self.blocked.discard(victim)
            for c,v in enumerate(victim):
                self.index[c,v].discard(victim)
                if not self.index[c,v]:del self.index[c,v]
            self.stats['witness_evictions']+=1
        return row in self.rows

    def add_cuts(self,cuts,check=lambda:None):
        self._check();added=0
        for raw in cuts:
            check();cut=tuple(revalidate_witness(self.problem,raw,target='gold'))
            if cut in self.cuts:continue
            candidates=None
            for literal in cut:
                current=self.index.get(literal,set());candidates=set(current) if candidates is None else candidates&current
                self.stats['witness_cut_checks']+=1
            check();self._check();self.cuts.add(cut);old=len(self.blocked);self.blocked.update(candidates or ())
            self.stats['witness_cut_invalidations']+=len(self.blocked)-old;added+=1
        return added

    def supports(self,domains,supported,check=lambda:None):
        self._check();domains=tuple(domains)
        if len(domains)!=self.problem.n or len(supported)!=self.problem.n or any(type(mask)is not int or mask<0 or mask&~sum(1<<v for v in row) for mask,row in zip(domains,self.binding[1])):
            raise ValueError('Witness query escapes original domains.')
        self.stats['witness_index_queries']+=1;candidates=set(self.rows)-self.blocked
        # Smallest indexed disjunction first. No witness/domain O(rows*cells)
        # scan; untouched cells with unrestricted masks require no intersection.
        restrictions=sorted(((sum(mask>>v&1 for v in row),c,mask) for c,(mask,row) in enumerate(zip(domains,self.binding[1]))
                             if mask!=sum(1<<v for v in row)))
        for _,cell,mask in restrictions:
            check();allowed=set()
            for code in self.binding[1][cell]:
                if mask>>code&1:allowed.update(self.index.get((cell,code),()));self.stats['witness_index_terms']+=1
            candidates&=allowed
            if not candidates:return 0
        result=list(supported);uses=[];gained=0
        for row in self.rows:
            if row not in candidates:continue
            check();count=sum(not mask>>code&1 for mask,code in zip(result,row))
            if count:
                uses.append(row);gained+=count
                for c,v in enumerate(row):result[c]|=1<<v
        check();self._check();supported[:]=result
        for row in uses:self.rows[row][1]+=1
        self.stats['witness_index_hits']+=len(uses);self.stats['witness_useful_supports']+=gained
        return gained

    def snapshot(self):
        self._check();body={'version':1,'fingerprint':self.binding[0],'domains':[list(row) for row in self.binding[1]],'boards':[list(row) for row in self.rows]}
        body['digest']=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();return body

    def import_snapshot(self,record,check=lambda:None):
        self._check()
        if type(record)is not dict or set(record)!={'version','fingerprint','domains','boards','digest'} or type(record['version'])is not int or record['version']!=1:
            raise ValueError('Invalid semantic witness file.')
        if len(json.dumps(record))>8*1024*1024 or record['fingerprint']!=self.binding[0] or record['domains']!=[list(row) for row in self.binding[1]]:
            raise ValueError('Semantic witness input differs.')
        body={k:v for k,v in record.items() if k!='digest'}
        if record['digest']!=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest() or type(record['boards'])is not list or len(record['boards'])>self.limit:
            raise ValueError('Semantic witness bounds/integrity differ.')
        # Validate all first: a malformed suffix cannot partially import a file.
        checked=[]
        for row in record['boards']:check();checked.append(self._validate(row))
        check();self._check()
        for row in checked:self.add(row)
        self.stats['witness_loaded']+=len(checked);return len(checked)

    def save(self,path):
        from io_utils import atomic_json
        atomic_json(Path(path),self.snapshot())

    def load(self,path,check=lambda:None):
        with Path(path).open('rb') as stream:raw=stream.read(8*1024*1024+1)
        if len(raw)>8*1024*1024:raise ValueError('Semantic witness file too large.')
        return self.import_snapshot(json.loads(raw),check)

    def transport(self,codes,cell_map,side_offsets,check=lambda:None,*,piece_map=None,piece_offsets=None):
        """Verify a proper oriented seam/domain automorphism before transport."""
        self._check();n=self.problem.n;cells=tuple(cell_map);offsets=tuple(side_offsets)
        if len(cells)!=n or any(type(c)is not int for c in cells) or set(cells)!=set(range(n)) or len(offsets)!=n or any(type(v)is not int or not 0<=v<3 for v in offsets):
            raise ValueError('Invalid witness symmetry bijection.')
        pieces=tuple(range(n)) if piece_map is None else tuple(piece_map);rotations=(0,)*n if piece_offsets is None else tuple(piece_offsets)
        if len(pieces)!=n or any(type(p)is not int for p in pieces) or set(pieces)!=set(range(n)) or len(rotations)!=n or any(type(r)is not int or not 0<=r<3 for r in rotations):raise ValueError('Invalid physical symmetry bijection.')
        def segments(tile,rotation):return tuple(sorted(tuple(sorted(((side+rotation)%3,position) for side,position in ends)) for ends in self.problem.tiles[tile]['segments']))
        for tile,mapped_tile in enumerate(pieces):
            check()
            if segments(tile,0)!=segments(mapped_tile,rotations[tile]):raise ValueError('Physical symmetry changes raw segment pairings.')
            self.stats['witness_symmetry_checks']+=1
        def image(code,cell):return 3*pieces[code//3]+(code%3+rotations[code//3]+offsets[cell])%3
        for cell in range(n):
            check()
            mapped={image(code,cell) for code in self.binding[1][cell]}
            if mapped!=set(self.binding[1][cells[cell]]):raise ValueError('Symmetry does not preserve original domains.')
            for side in range(3):
                other=int(self.problem.neighbors[cell,side]);target=(side+offsets[cell])%3
                expected=-1 if other<0 else cells[other]
                if int(self.problem.neighbors[cells[cell],target])!=expected:raise ValueError('Symmetry does not preserve board seams.')
                if other>=0 and int(self.problem.sides[cells[cell],target])!=(int(self.problem.sides[cell,side])+offsets[other])%3:
                    raise ValueError('Symmetry does not preserve oriented seams.')
                self.stats['witness_symmetry_checks']+=1
        row=self._validate(codes);mapped=[None]*n
        for cell,code in enumerate(row):mapped[cells[cell]]=image(code,cell)
        answer=self.add(mapped,check);self.stats['witness_transported']+=int(answer);return answer

    def expand_piece_symmetries(self,codes,check=lambda:None,*,limit=4):
        """Bounded exact identical-physical-piece transpositions in the service.

        Endpoint masks alone do not establish strand equality. Every proposed
        mapping is checked against complete raw segment pairings and all cell
        domains by transport(), followed by independent full-board validation.
        """
        if type(limit)is not int or not 0<=limit<=16:raise ValueError('Invalid witness transport budget.')
        n=self.problem.n;groups={};added=0;attempted=0
        for tile,item in enumerate(self.problem.tiles):
            signature=tuple(sorted(tuple(sorted(tuple(p) for p in ends)) for ends in item['segments']))
            groups.setdefault(signature,[]).append(tile)
        for group in groups.values():
            for index,tile in enumerate(group):
                for other in group[index+1:]:
                    check()
                    if attempted==limit:return added
                    attempted+=1;mapping=list(range(n));mapping[tile],mapping[other]=other,tile
                    # Domain-asymmetric fixed pieces are not automorphisms.
                    if any({3*mapping[c//3]+c%3 for c in row}!=set(row) for row in self.binding[1]):continue
                    added+=self.transport(codes,range(n),(0,)*n,check,piece_map=mapping)
        return added
