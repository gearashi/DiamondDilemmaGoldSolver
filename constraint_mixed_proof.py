"""Scoped proof DAGs combining independently checked finite-domain theories.

Every node proves a forbidden domain conjunction under one immutable puzzle,
root and objective. Hashes identify content; semantic checking supplies trust.
Checked lemmas can be reused between certificates in the same process. RUP
nodes combine Hall, proper-loop, parity and integer-inequality lemmas with the
reconstructed edge CNF. Caps never justify pruning.
"""
import json,time
from collections import OrderedDict
from constraint_proof import MaskProofContext,RUPChecker,BoundedDPLL,ProofLimit,MAX_PROOF_BYTES,_hash
from constraint_pruning import _tile_choices,values
from constraint_explanations import failure_witness
from constraint_knowledge import revalidate_witness
from constraint_algebra import AffinePruner,affine_equations
from constraint_pb import primitive_rows,derive_rows,verify_derivation,is_integer_row

FORMAT='diamond-mixed-mask-v1'
MAX_NODES=256
MAX_INTEGER_BITS=4096


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value,(tuple,list)) else value


class MixedProofContext(MaskProofContext):
    def __init__(self,problem,initial,target='gold'):
        super().__init__(problem,initial,target)
        self.nodes=OrderedDict();self.cursors=OrderedDict();self._gold_sources=OrderedDict()
        self.stats.update(mixed_nodes_checked=0,mixed_nodes_reused=0,mixed_proofs=0,
                          mixed_hall_nodes=0,mixed_loop_nodes=0,mixed_affine_nodes=0,mixed_pb_nodes=0)
        self.affine=AffinePruner(problem)

    def _current_binding(self):
        # Equal edge masks do not imply equal strand pairings. Cached Gold
        # lemmas must also bind the actual segment and seam-port wiring.
        return (super()._current_binding(),_hash(self.problem.tiles),
                tuple(tuple(map(int,row)) for row in self.problem.neighbors),
                tuple(tuple(map(int,row)) for row in self.problem.sides))

    def close(self):
        self.nodes.clear();self.cursors.clear();self._gold_sources.clear();super().close()

    def add_gold_witnesses(self,records):
        if self.target!='gold':return
        for placements in records:
            key=tuples(placements)
            self._gold_sources[key]=None
            while len(self._gold_sources)>MAX_NODES:self._gold_sources.popitem(last=False)

    def decode(self,record):
        return self.normalize([(c,int(mask,16)) for c,mask in record])

    def domains(self,record):
        result=list(self.initial)
        for c,mask in record:result[c]=mask
        return result

    def conclusion(self,record):
        return tuple(lit for cell,mask in record for code,lit in self.variables[cell].items()
                     if not mask>>code&1)

    def make_node(self,rule,record,data,parents=()):
        record=self.normalize(record)
        body={'rule':rule,'record':[[c,hex(mask)] for c,mask in record],
              'parents':list(parents),'data':json.loads(json.dumps(data))}
        return {'id':_hash([self.binding(()),body]),**body}

    def _accept(self,node,check,*,incremental=False,ticks=256):
        self._check();check()
        if not isinstance(node,dict) or set(node)!={'id','rule','record','parents','data'}:return False
        body={key:value for key,value in node.items() if key!='id'}
        if node['id']!=_hash([self.binding(()),body]):return False
        encoded=json.dumps(node,sort_keys=True,separators=(',',':'))
        cached=self.nodes.get(node['id'])
        if cached is not None:
            if cached[0]!=encoded:return False
            self.nodes.move_to_end(node['id']);self.stats['mixed_nodes_reused']+=1
            return True
        try:
            record=self.decode(node['record']);current=self.domains(record)
            parents=node['parents'];data=node['data'];rule=node['rule']
            if (not isinstance(parents,list) or len(parents)>MAX_NODES or len(set(parents))!=len(parents)
                    or any(not isinstance(key,str) or key not in self.nodes for key in parents)):return False
            if rule not in ('rup','weaken') and parents:return False
            if not isinstance(data,dict):return False
            if rule=='hall':
                if set(data)!={'cells'}:return False
                cells=data['cells']
                if (not isinstance(cells,list) or not cells or len(cells)>self.problem.n
                        or any(type(c)is not int or not 0<=c<self.problem.n for c in cells)
                        or len(set(cells))!=len(cells)):return False
                union=0
                for c in cells:check();union|=_tile_choices(current[c])
                valid=union.bit_count()<len(cells)
            elif rule=='loop':
                if set(data)!={'placements'} or self.target!='gold':return False
                witness=revalidate_witness(self.problem,data['placements'],target=self.target)
                valid=all(current[c] == 1<<code for c,code in witness)
            elif rule=='affine':
                if set(data)!={'equations'} or not self.affine.enabled:return False
                equations=data['equations']
                if not isinstance(equations,list) or not 1<=len(equations)<=1024:return False
                total=rhs=0
                for equation in equations:
                    check()
                    if (not isinstance(equation,list) or len(equation)!=3
                            or any(type(x)is not int for x in equation)):return False
                    cell,mask,value=equation
                    if not 0<=cell<self.problem.n or not 0<=mask<1<<33 or value not in (0,1):return False
                    for code in values(current[cell]):
                        check()
                        if (self.affine.patterns[code]&mask).bit_count()%2!=value:return False
                    global_mask=0
                    for port in values(mask):global_mask^=1<<self.affine.coordinates[cell][port]
                    total^=global_mask;rhs^=value
                valid=total==0 and rhs==1
            elif rule=='pb':
                if set(data)!={'cells','steps','row'}:return False
                cells=data['cells'];steps=tuples(data['steps']);index=data['row']
                if (not isinstance(cells,list) or not 1<=len(cells)<=8 or len(set(cells))!=len(cells)
                        or any(type(c)is not int or not 0<=c<self.problem.n for c in cells)
                        or not isinstance(steps,tuple) or len(steps)>128 or type(index)is not int):return False
                for step in steps:
                    check()
                    if (not isinstance(step,tuple) or len(step)!=5
                            or any(type(x)is not int or x.bit_length()>MAX_INTEGER_BITS for x in step[:4])
                            or not is_integer_row(step[4],max_bits=MAX_INTEGER_BITS)):return False
                base=primitive_rows(self.problem,tuple((c,current[c]) for c in cells))
                if not verify_derivation(base,steps,check):return False
                rows=list(dict.fromkeys(base))+[step[-1] for step in steps]
                if not 0<=index<len(rows):return False
                terms,bound=rows[index];by_cell={}
                for cell,code,coefficient in terms:
                    by_cell.setdefault(cell,{})[code]=coefficient
                lower=0
                for cell,coefficients in by_cell.items():
                    check();lower+=min(coefficients.get(code,0) for code in values(current[cell]))
                valid=lower>bound
            elif rule=='weaken':
                if data or len(parents)!=1:return False
                prior=self.nodes[parents[0]][2]
                valid=all(not current[c]&~mask for c,mask in prior)
            elif rule=='rup':
                if set(data)!={'proof'}:return False
                base=self.clauses(record)+tuple(self.nodes[key][1] for key in parents)
                if incremental:
                    from constraint_proof_cursor import ResumableRUPChecker
                    cursor=self.cursors.get(node['id'])
                    if cursor is None:
                        cursor=ResumableRUPChecker(base,data['proof'])
                        self.cursors[node['id']]=cursor
                        while len(self.cursors)>8:self.cursors.popitem(last=False)
                    before=cursor.stats.copy()
                    try:valid=cursor.advance(check,ticks=ticks)
                    finally:
                        self.stats['proof_cursor_visits']+=cursor.stats['proof_clause_visits']-before['proof_clause_visits']
                        self.stats['proof_cursor_resumptions']+=cursor.stats['proof_resumptions']-before['proof_resumptions']
                    if cursor.limited:self.last_verification_status='limited';return None
                    if valid is None:return None
                else:
                    checker=RUPChecker(base);valid=checker.verify(data['proof'],check)
                    self.stats['proof_cursor_visits']+=checker.stats['proof_clause_visits']
            else:return False
        except (ValueError,TypeError,KeyError,IndexError,OverflowError):return False
        if not valid:return False
        self.nodes[node['id']]=(encoded,self.conclusion(record),record)
        self.nodes.move_to_end(node['id']);self.stats['mixed_nodes_checked']+=1
        if rule in ('hall','loop','affine','pb'):self.stats['mixed_'+rule+'_nodes']+=1
        self.cursors.pop(node['id'],None)
        while len(self.nodes)>MAX_NODES:self.nodes.popitem(last=False)
        return True

    def bundle(self,root):
        output=[];seen=set()
        def visit(key):
            if key in seen:return
            if key not in self.nodes:raise ProofLimit('Proof dependency evicted.')
            node=json.loads(self.nodes[key][0])
            for parent in node['parents']:visit(parent)
            seen.add(key);output.append(node)
        visit(root);record=self.nodes[root][2]
        result={'proof_format':FORMAT,**self.binding(record),'nodes':output,'root':root}
        if len(json.dumps(result))>MAX_PROOF_BYTES:raise ProofLimit('Mixed proof byte cap.')
        return result

    def _verify(self,certificate,check,incremental,ticks):
        self._check();self.last_verification_status='invalid'
        if not isinstance(certificate,dict) or set(certificate)!={'proof_format','fingerprint','target','initial','record','base_sha256','nodes','root'}:return False
        try:
            if certificate['proof_format']!=FORMAT or len(json.dumps(certificate))>MAX_PROOF_BYTES:return False
            if not isinstance(certificate['root'],str):return False
            record=self.decode(certificate['record']);nodes=certificate['nodes']
            if not isinstance(nodes,list) or not 1<=len(nodes)<=MAX_NODES:return False
            if len({node['id'] for node in nodes})!=len(nodes):return False
        except (ValueError,TypeError,KeyError,OverflowError):return False
        if not self.prepare(check,seconds=1):
            self.last_verification_status='pending';return None if incremental else False
        if any(certificate.get(k)!=v for k,v in self.binding(record).items()):return False
        for node in nodes:
            valid=self._accept(node,check,incremental=incremental,ticks=ticks)
            if valid is None:
                if self.last_verification_status!='limited':self.last_verification_status='pending'
                return None
            if not valid:return False
        root=certificate['root']
        if root not in self.nodes or self.nodes[root][2]!=record:return False
        self.last_verification_status='valid';self.stats['mixed_proofs']+=1
        return True

    def verify(self,certificate,check=lambda:None):
        if isinstance(certificate,dict) and certificate.get('proof_format')!=FORMAT:
            return super().verify(certificate,check)
        return self._verify(certificate,check,False,2000000)

    def verify_incremental(self,certificate,check=lambda:None,*,ticks=256):
        if isinstance(certificate,dict) and certificate.get('proof_format')!=FORMAT:
            return super().verify_incremental(certificate,check,ticks=ticks)
        return self._verify(certificate,check,True,ticks)

    def _affine_leaf(self,record,check):
        if not self.affine.enabled or self.problem.n>16:return None
        current=self.domains(record);basis={};equations=[]
        for cell,domain in enumerate(current):
            check()
            for mask,rhs in affine_equations((self.affine.patterns[code] for code in values(domain)),33):
                if len(equations)>=1024:return None
                identity=1<<len(equations);equations.append([cell,mask,rhs])
                global_mask=0
                for port in values(mask):global_mask^=1<<self.affine.coordinates[cell][port]
                equation=(global_mask<<1)|rhs
                while equation>>1:
                    check();pivot=equation.bit_length()-1
                    if pivot not in basis:basis[pivot]=(equation,identity);break
                    other,causes=basis[pivot];equation^=other;identity^=causes
                else:
                    if equation==1:
                        return self.make_node('affine',record,{'equations':[equations[i] for i in values(identity)]})
        return None

    def _pb_leaf(self,record,check):
        if self.problem.n>8:return None
        current=self.domains(record);scope=tuple(enumerate(current))
        base=primitive_rows(self.problem,scope);rows,steps,_=derive_rows(base,check,max_pairs=128,max_new=32)
        for index,(terms,bound) in enumerate(rows):
            check();by_cell={}
            for cell,code,coefficient in terms:by_cell.setdefault(cell,{})[code]=coefficient
            if sum(min(coefficients.get(code,0) for code in values(current[cell]))
                   for cell,coefficients in by_cell.items())>bound:
                # derive_rows returns rows in this exact base+append order.
                return self.make_node('pb',record,{'cells':list(range(self.problem.n)),'steps':steps,'row':index})
        return None

    def prove(self,record,check=lambda:None,*,seconds=.1,conflicts=10000):
        deadline=time.monotonic()+seconds
        def bounded():
            check()
            if time.monotonic()>=deadline:raise ProofLimit('Mixed proof time cap.')
        try:
            if not self.prepare(bounded,seconds=max(0,deadline-time.monotonic())):return None
            record=self.normalize(record);current=self.domains(record)
            witness=failure_witness([_tile_choices(mask) for mask in current],bounded)
            candidates=[]
            if witness is not None:
                local=self.normalize([(c,current[c]) for c in witness])
                candidates.append(self.make_node('hall',local,{'cells':list(witness)}))
            if self.target=='gold':
                placements=tuple((c,mask.bit_length()-1) for c,mask in enumerate(current) if mask.bit_count()==1)
                for placed in [placements,*list(self._gold_sources)]:
                    bounded()
                    try:
                        witness=revalidate_witness(self.problem,placed,target=self.target)
                        local=self.normalize([(c,1<<code) for c,code in witness])
                    except ValueError:continue
                    candidates.append(self.make_node('loop',local,{'placements':witness}))
            for node in candidates:
                if self._accept(node,bounded):
                    prior=self.nodes[node['id']][2]
                    if all(not current[c]&~mask for c,mask in prior):
                        if prior==record:return self.bundle(node['id'])
                        weakened=self.make_node('weaken',record,{},[node['id']])
                        if self._accept(weakened,bounded):return self.bundle(weakened['id'])
            for create in (self._affine_leaf,self._pb_leaf):
                node=create(record,bounded)
                if node is not None and self._accept(node,bounded):return self.bundle(node['id'])
            parents=list(self.nodes)[-128:]
            clauses=self.clauses(record)+tuple(self.nodes[key][1] for key in parents)
            producer=BoundedDPLL(clauses,max_decisions=min(512,max(1,conflicts)))
            try:answer=producer.solve(bounded)
            finally:
                for key,value in producer.stats.items():self.stats[key]+=value
            if answer:return None
            proof=list(producer.proof)
            if not any(line.strip()=='0' for line in proof):proof.append('0')
            node=self.make_node('rup',record,{'proof':proof},parents)
            if not self._accept(node,bounded):raise RuntimeError('Mixed producer failed independent checking.')
            return self.bundle(node['id'])
        except ProofLimit:self.stats['proof_unknown']+=1;return None
