"""Memory-bounded exact CUDA domain-trail search (round24 research engine).

Each lane owns one disjoint domain cube. Domains, inference scans, reversible
removals, connectivity unions and unfinished choices remain on the device.
The engine intentionally caps replicas by its explicit worst-case state size.
It is separate from the compact prefix engine used for 131072-lane searches.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from validator import reverse_mask

ROOT=Path(__file__).resolve().parent
META=29
# meta: state,depth,trail_n,scan_cell,scan_code,changed,phase,clause_scan,
#       clauses,assign_n,union_n,nodes,removals,checks,backjumps,learned,
#       strand_joins,clauses_applied,solutions,clause_head

class DomainCUDA:
    SCHEMA='diamond-domain-cuda-v1'
    def __init__(self,problem,cubes=None,*,target='gold',max_memory_bytes=512*1024*1024,
                 learning=True,incremental_strands=True):
        if target not in ('gold','edges'):raise ValueError('Invalid domain-engine target.')
        if type(max_memory_bytes) is not int or max_memory_bytes<1:raise ValueError('Positive memory limit required.')
        self.problem,self.target=problem,target;self.n=problem.n;self.nc=3*self.n
        if not 1<=self.n<=160:raise ValueError('Domain CUDA supports1..160 cells.')
        self.words=(self.nc+31)//32;self.reason_words=(self.n+31)//32
        self.learning=bool(learning);self.incremental_strands=bool(incremental_strands)
        self.binding={'fingerprint':problem.fingerprint,'target':target,'original':self._original(),
                      'learning':self.learning,'incremental_strands':self.incremental_strands}
        if cubes is None:cubes=[tuple(int(value,16) for value in self.binding['original'])]
        self.cubes=tuple(self._validate_cube(cube) for cube in cubes)
        self.lanes=len(self.cubes)
        if not self.lanes:raise ValueError('At least one domain cube is required.')
        for i,left in enumerate(self.cubes):
            for right in self.cubes[:i]:
                if all(a&b for a,b in zip(left,right)):raise ValueError('Domain CUDA cubes overlap.')
        n,w,rw,nc=self.n,self.words,self.reason_words,self.nc
        fields=[('meta',META),('domain',n*w),('trail',n*nc),('reason',n*rw),
                ('frame_cell',n),('frame_code',n),('frame_remaining',n*w),('frame_trail',n),
                ('frame_union',n),('frame_assign',n),('frame_reason',n*n*rw),('frame_failure',n*rw),
                ('board',n),('owner',n),('assign_log',n),('parent',n*16),('size',n*16),
                ('union_child',n*16),('union_parent',n*16),('union_size',n*16),
                ('clause_length',32),('clause_cell',32*n),('clause_code',32*n)]
        self.offsets={};cursor=0
        for name,size in fields:self.offsets[name]=cursor;cursor+=size
        self.rows=cursor;self.state_bytes=cursor*self.lanes*4
        if self.state_bytes>max_memory_bytes:
            maximum=max_memory_bytes//(cursor*4)
            raise ValueError(f'Domain-trail state needs {self.state_bytes:,} bytes; this budget permits at most {maximum} lanes (requested {self.lanes}).')
        import cupy as cp
        self.cp=cp
        free,_=cp.cuda.runtime.memGetInfo()
        if self.state_bytes>free*0.75:raise ValueError('Domain-trail state exceeds75% of currently free VRAM; reduce lanes.')
        data=self._initial_state();self.state=cp.asarray(data)
        tables=self._tables();self.tables=cp.asarray(tables)
        constants={'N':n,'NC':nc,'W':w,'RW':rw,'P':16*n,'C':32,'ROWS':cursor,
                   'GOLD':int(target=='gold'),'LEARNING':int(learning),'INCREMENTAL_STRANDS':int(incremental_strands),
                   'TOTAL_SEGMENTS':problem.total_segments,**{key.upper():value for key,value in self.offsets.items()}}
        prefix='\n'.join(f'#define {name} {value}' for name,value in constants.items())+'\n'
        self.module=cp.RawModule(code=prefix+(ROOT/'domain_search_round24.cu').read_text(),options=('--std=c++17',),
                                 name_expressions=('domain_search','domain_ack'))
        self.kernel=self.module.get_function('domain_search');self.ack_kernel=self.module.get_function('domain_ack')
        name=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'];self.device=name.decode() if isinstance(name,bytes) else str(name)
        self.delivered=[]
    def _original(self):return [hex(sum(1<<int(code) for code in row)) for row in self.problem.domains]
    def _validate_cube(self,cube):
        if len(cube)!=self.n or any(type(value) is not int or value<=0 or value>>(3*self.n) for value in cube):raise ValueError('Invalid positive orientation-domain cube.')
        original=[int(value,16) for value in self.binding['original']]
        if any(value&~allowed for value,allowed in zip(cube,original)):raise ValueError('Cube exceeds original problem domains.')
        return tuple(cube)
    def _initial_state(self):
        data=np.zeros((self.rows,self.lanes),np.uint32)
        for lane,cube in enumerate(self.cubes):
            for cell,mask in enumerate(cube):
                for word in range(self.words):data[self.offsets['domain']+cell*self.words+word,lane]=(mask>>(32*word))&0xffffffff
        data[self.offsets['board']:self.offsets['board']+self.n]=0xffffffff
        data[self.offsets['owner']:self.offsets['owner']+self.n]=0xffffffff
        data[self.offsets['parent']:self.offsets['parent']+self.n*16]=np.arange(self.n*16,dtype=np.uint32)[:,None]
        data[self.offsets['size']:self.offsets['size']+self.n*16]=1
        return data
    def _tables(self):
        # Fixed offsets: neighbors[3N], sides[3N], masks[9N], reversed[9N],
        # segment counts[N], endpoint->segment[NC*33], compatibility[9*NC*W].
        ports=np.full((self.nc,33),0xffffffff,np.uint32)
        for code,rows in enumerate(self.problem._segment_ports):ports[code]=np.asarray(rows,dtype=np.int32).ravel().view(np.uint32)
        masks=np.asarray(self.problem.masks,np.uint32)
        reverse=np.asarray([[reverse_mask(value) for value in row] for row in masks],np.uint32)
        compatible=np.zeros((9,self.nc,self.words),np.uint32)
        for side in range(3):
            for other_side in range(3):
                supports=masks[:,side,None]==reverse[None,:,other_side]
                packed=np.packbits(supports,axis=1,bitorder='little')
                storage=np.zeros((self.nc,self.words*4),np.uint8)
                storage[:,:packed.shape[1]]=packed
                compatible[side*3+other_side]=storage.view('<u4')
        return np.concatenate((self.problem.neighbors.astype(np.int32).ravel().view(np.uint32),self.problem.sides.astype(np.int32).ravel().view(np.uint32),
            masks.ravel(),reverse.ravel(),np.array(self.problem._segment_counts,np.uint32),ports.ravel(),compatible.ravel()))
    def step(self,work=1024):
        if type(work) is not int or not 1<=work<=1000000:raise ValueError('Work quantum must be1..1000000.')
        self.kernel(((self.lanes+63)//64,),(64,),(self.state,self.tables,np.int32(self.lanes),np.int32(work)))
        self.cp.cuda.get_current_stream().synchronize()
        return self.state[self.offsets['meta']].get().astype(int)
    def stats(self):
        m=self.state[:META].get().astype(np.uint64)
        return {'lanes':self.lanes,'state_bytes':self.state_bytes,'active':int(np.sum(m[0]==0)),
                'pending':int(np.sum(m[0]==1)),'exhausted':int(np.sum(m[0]==2)),
                **{name:int((m[index]+(m[20+index-11]<<np.uint64(32))).sum()) for name,index in [('decisions',11),('domain_removals',12),('support_checks',13),('backjumps',14),('learned_clauses',15),('strand_joins',16),('clause_units',17),('solutions',18)]}}
    def candidates(self):
        states=self.state[0].get();answers=[]
        pending=np.flatnonzero(states==1)
        if not len(pending):return answers
        domains=self.state[self.offsets['domain']:self.offsets['domain']+self.n*self.words,pending].get()
        for column,lane in enumerate(pending):
            codes=[]
            for cell in range(self.n):
                mask=sum(int(domains[cell*self.words+word,column])<<(32*word) for word in range(self.words))
                if mask.bit_count()!=1:raise RuntimeError('Pending CUDA candidate has a nonsingleton domain.')
                codes.append(mask.bit_length()-1)
            report=self.problem.validate(codes)
            accepted=report['valid'] or self.target=='edges' and report['unique_tiles'] and report['matched_edges']==len(self.problem.edges)
            if not accepted or any(not(self.cubes[lane][cell]&(1<<code)) for cell,code in enumerate(codes)):
                raise RuntimeError('Domain CUDA candidate failed independent validation.')
            answers.append({'lane':int(lane),'codes':codes,'validation':report})
        return answers
    def acknowledge(self,lanes):
        # Enumeration acknowledgement records returned solutions; normal app
        # solving stops at its first independently verified Gold candidate.
        lanes=list(lanes)
        pending={item['lane']:item for item in self.candidates()}
        if any(type(lane)is not int or lane not in pending for lane in lanes):raise ValueError('Only validated pending lanes may be acknowledged.')
        if len(set(lanes))!=len(lanes):raise ValueError('Duplicate acknowledgement.')
        for lane in lanes:self.delivered.append(pending[lane])
        ids=self.cp.asarray(lanes,dtype=np.int32)
        if lanes:self.ack_kernel(((len(lanes)+63)//64,),(64,),(self.state,ids,np.int32(self.lanes),np.int32(len(lanes))))
        self.cp.cuda.get_current_stream().synchronize()
    def _mask(self,data,lane,cell):
        offset=self.offsets['domain']+cell*self.words
        return sum(int(data[offset+word,lane])<<(32*word) for word in range(self.words))
    def remaining_domain_cubes(self):
        data=self.state.get();result=[]
        for lane in range(self.lanes):
            if data[0,lane]==2:continue
            current=[self._mask(data,lane,cell) for cell in range(self.n)]
            # phase3/4 is a failed or already-delivered current branch.
            phase=int(data[6,lane]);depth=int(data[1,lane]);trail=int(data[2,lane])
            if data[0,lane]==1 or phase not in (3,4):
                if all(current):result.append(tuple(current))
            for level in range(depth-1,-1,-1):
                mark=int(data[self.offsets['frame_trail']+level,lane])
                for pos in range(trail-1,mark-1,-1):
                    encoded=int(data[self.offsets['trail']+pos,lane]);cell,code=divmod(encoded,self.nc);current[cell] |= 1<<code
                trail=mark;cell=int(data[self.offsets['frame_cell']+level,lane])
                offset=self.offsets['frame_remaining']+level*self.words
                remaining=sum(int(data[offset+word,lane])<<(32*word) for word in range(self.words))
                if remaining:
                    sibling=current.copy();sibling[cell]=remaining;result.append(tuple(sibling))
        return result
    def checkpoint(self):
        # Native arrays are retained exactly; portable cube transport separately
        # preserves remaining coverage while intentionally losing engine scratch.
        data=self.state.get()
        return {'schema':self.SCHEMA,'binding':self.binding,'cubes':[[hex(v) for v in cube] for cube in self.cubes],
                'state':data,'state_sha256':hashlib.sha256(data.tobytes()).hexdigest(),'delivered':list(self.delivered),'rows':self.rows}
    def restore(self,checkpoint):
        if checkpoint.get('schema')!=self.SCHEMA or checkpoint.get('binding')!=self.binding or checkpoint.get('cubes')!=[[hex(v) for v in cube] for cube in self.cubes]:raise ValueError('Domain checkpoint scope differs.')
        state=checkpoint.get('state')
        if not isinstance(state,np.ndarray) or state.dtype!=np.uint32 or state.shape!=(self.rows,self.lanes):raise ValueError('Malformed domain checkpoint arrays.')
        if checkpoint.get('state_sha256')!=hashlib.sha256(state.tobytes()).hexdigest():raise ValueError('Domain checkpoint state checksum differs.')
        # A checksum is integrity protection, not a mathematical proof. Local
        # native checkpoints are trusted process state; portable imports use
        # explicit cubes and reconstruct all deductions.
        if np.any(state[0]>2) or np.any(state[1]>self.n) or np.any(state[2]>self.n*self.nc) or np.any(state[9]>self.n) or np.any(state[10]>self.n*16):raise ValueError('Invalid domain checkpoint counters.')
        if np.any(state[3]>self.n) or np.any(state[4]>self.nc) or np.any(state[6]>4) or np.any(state[7]>32) or np.any(state[8]>32) or np.any(state[19]>=32):raise ValueError('Invalid domain checkpoint scan state.')
        for name,length,maximum,sentinel in [('board',self.n,self.nc,True),('owner',self.n,self.n,True),('parent',self.n*16,self.n*16,False)]:
            values=state[self.offsets[name]:self.offsets[name]+length]
            if np.any((values>=maximum)&(values!=0xffffffff if sentinel else True)):raise ValueError('Invalid domain checkpoint '+name)
        for lane in range(self.lanes):
            trail=state[self.offsets['trail']:self.offsets['trail']+int(state[2,lane]),lane]
            if np.any(trail>=self.n*self.nc):raise ValueError('Invalid domain removal trail.')
            for level in range(int(state[1,lane])):
                if state[self.offsets['frame_cell']+level,lane]>=self.n or state[self.offsets['frame_code']+level,lane]>=self.nc:raise ValueError('Invalid domain decision frame.')
                if state[self.offsets['frame_trail']+level,lane]>state[2,lane] or state[self.offsets['frame_assign']+level,lane]>state[9,lane] or state[self.offsets['frame_union']+level,lane]>state[10,lane]:raise ValueError('Invalid domain rollback marker.')
        # Mechanical state invariants are checked even for trusted local
        # checkpoints. A rehashed malformed rollback pointer must not launch a
        # device out-of-bounds write or an endless disjoint-set find loop.
        for lane,cube in enumerate(self.cubes):
            domains=tuple(self._mask(state,lane,cell) for cell in range(self.n))
            if any(mask&~root for mask,root in zip(domains,cube)):
                raise ValueError('Domain checkpoint escapes its owned cube.')
            status,phase,depth,trail_n=(int(state[index,lane]) for index in (0,6,1,2))
            if status==0 and phase not in (3,4) and not all(domains):
                raise ValueError('Active nonfailed checkpoint has an empty domain.')
            removed=[0]*self.n;seen=set()
            for encoded in state[self.offsets['trail']:self.offsets['trail']+trail_n,lane]:
                encoded=int(encoded)
                if encoded in seen:raise ValueError('Duplicate active domain removal.')
                seen.add(encoded);cell,code=divmod(encoded,self.nc);removed[cell]|=1<<code
            if any(lost!=(root&~current) for lost,root,current in zip(removed,cube,domains)):
                raise ValueError('Domain removal trail does not match the owned cube.')
            parent=state[self.offsets['parent']:self.offsets['parent']+self.n*16,lane]
            size=state[self.offsets['size']:self.offsets['size']+self.n*16,lane]
            if np.any(size<1) or np.any(size>self.n*16):raise ValueError('Invalid connectivity component size.')
            done=set()
            for start in range(self.n*16):
                if start in done:continue
                path=set();node=start
                while node not in done:
                    if node in path:raise ValueError('Cyclic connectivity parents in checkpoint.')
                    path.add(node);next_node=int(parent[node])
                    if next_node==node:break
                    node=next_node
                done.update(path)
            union_n=int(state[10,lane])
            children=state[self.offsets['union_child']:self.offsets['union_child']+union_n,lane]
            parents=state[self.offsets['union_parent']:self.offsets['union_parent']+union_n,lane]
            sizes=state[self.offsets['union_size']:self.offsets['union_size']+union_n,lane]
            if (np.any(children>=self.n*16) or np.any(parents>=self.n*16) or np.any(children==parents)
                    or np.any(sizes<1) or np.any(sizes>self.n*16)):
                raise ValueError('Invalid connectivity rollback record.')
            assigned_n=int(state[9,lane])
            assigned=[int(v) for v in state[self.offsets['assign_log']:self.offsets['assign_log']+assigned_n,lane]]
            if len(set(assigned))!=len(assigned) or any(cell>=self.n for cell in assigned):
                raise ValueError('Invalid assignment rollback log.')
            boards=[int(v) for v in state[self.offsets['board']:self.offsets['board']+self.n,lane]]
            owners=[int(v) for v in state[self.offsets['owner']:self.offsets['owner']+self.n,lane]]
            if set(assigned)!={cell for cell,code in enumerate(boards) if code!=0xffffffff}:
                raise ValueError('Assignment log and board differ.')
            expected_owners=[0xffffffff]*self.n
            for cell in assigned:
                code=boards[cell]
                if not cube[cell]&(1<<code) or expected_owners[code//3]!=0xffffffff:
                    raise ValueError('Assignment escapes cube or repeats a physical piece.')
                expected_owners[code//3]=cell
                if status!=2 and phase not in (3,4) and domains[cell]!=(1<<code):
                    raise ValueError('Assigned cell and live domain differ.')
            if owners!=expected_owners:raise ValueError('Physical-piece ownership differs from board.')
            # Restoring parent domains from the active removal trail also
            # verifies every retained sibling mask and its selected code.
            restored=list(domains);cursor=trail_n;previous_mark=-1
            for level in range(depth):
                mark=int(state[self.offsets['frame_trail']+level,lane])
                if mark<previous_mark:raise ValueError('Decision rollback markers are not nested.')
                previous_mark=mark
            for level in range(depth-1,-1,-1):
                mark=int(state[self.offsets['frame_trail']+level,lane])
                for pos in range(cursor-1,mark-1,-1):
                    cell,code=divmod(int(state[self.offsets['trail']+pos,lane]),self.nc);restored[cell]|=1<<code
                cursor=mark;cell=int(state[self.offsets['frame_cell']+level,lane]);code=int(state[self.offsets['frame_code']+level,lane])
                offset=self.offsets['frame_remaining']+level*self.words
                remaining=sum(int(state[offset+word,lane])<<(32*word) for word in range(self.words))
                if remaining&~restored[cell] or remaining&(1<<code) or not restored[cell]&(1<<code):
                    raise ValueError('Decision choices escape the restored parent domain.')
            if status==1:
                if any(mask.bit_count()!=1 for mask in domains):raise ValueError('Pending checkpoint is not a complete candidate.')
                codes=[mask.bit_length()-1 for mask in domains];report=self.problem.validate(codes)
                if not (report['valid'] or self.target=='edges' and report['unique_tiles'] and report['matched_edges']==len(self.problem.edges)):
                    raise ValueError('Pending checkpoint candidate failed independent validation.')
        delivered=checkpoint.get('delivered',[])
        if not isinstance(delivered,list):raise ValueError('Malformed delivered-candidate history.')
        verified=[];seen=set()
        for item in delivered:
            if not isinstance(item,dict) or type(item.get('lane'))is not int or not 0<=item['lane']<self.lanes:
                raise ValueError('Malformed delivered lane.')
            codes=item.get('codes');lane=item['lane']
            if not isinstance(codes,list) or len(codes)!=self.n or any(type(code)is not int or code<0 or code>=self.nc for code in codes):
                raise ValueError('Malformed delivered candidate.')
            if tuple(codes) in seen or any(not self.cubes[lane][cell]&(1<<code) for cell,code in enumerate(codes)):
                raise ValueError('Delivered candidate is repeated or outside its lane.')
            report=self.problem.validate(codes)
            if not (report['valid'] or self.target=='edges' and report['unique_tiles'] and report['matched_edges']==len(self.problem.edges)):
                raise ValueError('Delivered candidate failed independent validation.')
            seen.add(tuple(codes));verified.append({'lane':lane,'codes':list(codes),'validation':report})
        self.state[:]=self.cp.asarray(state);self.delivered=verified
    def close(self):self.state=self.tables=self.kernel=self.ack_kernel=self.module=None
