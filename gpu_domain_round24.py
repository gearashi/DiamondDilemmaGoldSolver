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
    schema=SCHEMA  # Legacy host-only checkpoint fixtures retain the v1 layout.
    event_propagation=False
    precise_reasons=False
    support_reuse=False
    support_metrics=False
    watched_clauses=False
    clause_metrics=False
    global_matching=False
    cooperative=False
    proof_transport=False
    def __init__(self,problem,cubes=None,*,target='gold',max_memory_bytes=512*1024*1024,
                 learning=True,incremental_strands=True,joint_support=False,event_propagation=False,precise_reasons=False,reason_slots=None,support_reuse=False,support_metrics=False,support_slots=None,watched_clauses=False,clause_metrics=False,global_matching=False,cooperative=False,proof_transport=False):
        if target not in ('gold','edges'):raise ValueError('Invalid domain-engine target.')
        if type(max_memory_bytes) is not int or max_memory_bytes<1:raise ValueError('Positive memory limit required.')
        self.problem,self.target=problem,target;self.n=problem.n;self.nc=3*self.n
        if not 1<=self.n<=160:raise ValueError('Domain CUDA supports1..160 cells.')
        self.words=(self.nc+31)//32;self.reason_words=(self.n+31)//32
        self.learning=bool(learning);self.incremental_strands=bool(incremental_strands)
        if type(joint_support)is not bool:raise ValueError('joint_support must be boolean.')
        if type(event_propagation)is not bool:raise ValueError('event_propagation must be boolean.')
        if type(precise_reasons)is not bool:raise ValueError('precise_reasons must be boolean.')
        if reason_slots is not None and (not precise_reasons or type(reason_slots)is not int or not 8<=reason_slots<=2048 or reason_slots&(reason_slots-1)):
            raise ValueError('Reason slots require precise mode and a power of two from8 to2048.')
        from gpu_reasoning_round26 import validate_support_policy
        self.support_metrics,self.support_slots=validate_support_policy(support_reuse,support_metrics,support_slots)
        self.support_reuse=support_reuse
        if type(watched_clauses)is not bool or type(clause_metrics)is not bool:raise ValueError('Watched clauses/metrics must be boolean.')
        self.watched_clauses=watched_clauses;self.clause_metrics=bool(watched_clauses or clause_metrics)
        if type(global_matching)is not bool:raise ValueError("Global matching must be boolean.")
        self.global_matching=global_matching
        if type(cooperative)is not bool:raise ValueError("Cooperative execution must be boolean.")
        self.cooperative=cooperative
        if type(proof_transport)is not bool:raise ValueError("Proof transport must be boolean.")
        self.proof_transport=proof_transport
        self.precise_reasons=precise_reasons
        self.reason_slots=reason_slots if reason_slots is not None else min(2048,max(8,1<<(2*self.n*self.nc-1).bit_length()))
        self.joint_support=joint_support;self.event_propagation=event_propagation
        self.schema='diamond-domain-cuda-v3' if event_propagation else ('diamond-domain-cuda-v2' if joint_support else self.SCHEMA)
        if precise_reasons:self.schema='diamond-domain-cuda-v4'
        if self.support_metrics:self.schema='diamond-domain-cuda-v5'
        if self.clause_metrics:self.schema='diamond-domain-cuda-v6'
        if global_matching:self.schema='diamond-domain-cuda-v7'
        if proof_transport:self.schema='diamond-domain-cuda-v8'
        self.binding={'fingerprint':problem.fingerprint,'target':target,'original':self._original(),
                      'learning':self.learning,'incremental_strands':self.incremental_strands}
        if joint_support:self.binding['joint_support']=True
        if event_propagation:self.binding['event_propagation']=True
        if precise_reasons:self.binding.update(precise_reasons=True,reason_slots=self.reason_slots)
        if self.support_metrics:self.binding.update(support_metrics=True,support_reuse=support_reuse)
        if support_reuse:self.binding['support_slots']=self.support_slots
        if self.clause_metrics:self.binding.update(watched_clauses=watched_clauses,clause_metrics=True)
        if global_matching:self.binding['global_matching']=True
        if cooperative:self.binding['cooperative']=True
        if proof_transport:self.binding['proof_transport']=True
        if cubes is None:cubes=[tuple(int(value,16) for value in self.binding['original'])]
        self.cubes=tuple(self._validate_cube(cube) for cube in cubes)
        self.lanes=len(self.cubes)
        if not self.lanes:raise ValueError('At least one domain cube is required.')
        for i,left in enumerate(self.cubes):
            for right in self.cubes[:i]:
                if all(a&b for a,b in zip(left,right)):raise ValueError('Domain CUDA cubes overlap.')
        n,w,rw,nc=self.n,self.words,self.reason_words,self.nc
        from gpu_reasoning_round26 import domain_layout
        fields=domain_layout(n,joint_support=joint_support,event_propagation=event_propagation,precise_reasons=precise_reasons,
            reason_slots=reason_slots,support_reuse=support_reuse,support_metrics=support_metrics,support_slots=support_slots,
            watched_clauses=watched_clauses,clause_metrics=clause_metrics,global_matching=global_matching,proof_transport=proof_transport,cooperative=cooperative)
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
        data=self._initial_state();self.state=cp.asarray(data,order="F" if cooperative else "C")
        tables=self._tables();self.tables=cp.asarray(tables)
        constants={'N':n,'NC':nc,'W':w,'RW':rw,'P':16*n,'C':32,'ROWS':cursor,
                   'GOLD':int(target=='gold'),'LEARNING':int(learning),'INCREMENTAL_STRANDS':int(incremental_strands),
                   'TOTAL_SEGMENTS':problem.total_segments,'JOINT_SUPPORT':int(joint_support),'EVENT_PROPAGATION':int(event_propagation),'PRECISE_REASONS':int(precise_reasons),'RS':self.reason_slots,'PROOF_TRANSPORT':int(proof_transport),'COOPERATIVE':int(cooperative),'GLOBAL_MATCHING':int(global_matching),'WATCHED_CLAUSES':int(watched_clauses),'CLAUSE_METRICS':int(self.clause_metrics),'SUPPORT_REUSE':int(support_reuse),'SUPPORT_METRICS':int(self.support_metrics),'SS':self.support_slots,'TW':(n+31)//32,**{key.upper():value for key,value in self.offsets.items()}}
        prefix='\n'.join(f'#define {name} {value}' for name,value in constants.items())+'\n'
        self.module=cp.RawModule(code=prefix+(ROOT/'domain_search_round24.cu').read_text(),options=('--std=c++17',),
                                 name_expressions=('domain_search','domain_ack','domain_propagate','domain_share'))
        self.kernel=self.module.get_function('domain_search');self.ack_kernel=self.module.get_function('domain_ack')
        self.propagate_kernel=self.module.get_function('domain_propagate')
        self.share_kernel=self.module.get_function('domain_share')
        name=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'];self.device=name.decode() if isinstance(name,bytes) else str(name)
        self.delivered=[]
    def _original(self):return [hex(sum(1<<int(code) for code in row)) for row in self.problem.domains]
    def _validate_cube(self,cube):
        if len(cube)!=self.n or any(type(value) is not int or value<=0 or value>>(3*self.n) for value in cube):raise ValueError('Invalid positive orientation-domain cube.')
        original=[int(value,16) for value in self.binding['original']]
        if any(value&~allowed for value,allowed in zip(cube,original)):raise ValueError('Cube exceeds original problem domains.')
        return tuple(cube)
    def _initial_state(self,cubes=None):
        roots=self.cubes if cubes is None else tuple(cubes)
        data=np.zeros((self.rows,len(roots)),np.uint32)
        for lane,cube in enumerate(roots):
            for cell,mask in enumerate(cube):
                for word in range(self.words):data[self.offsets['domain']+cell*self.words+word,lane]=(mask>>(32*word))&0xffffffff
        if self.precise_reasons:data[self.offsets['precise_root']:self.offsets['precise_root']+self.n*self.words]=data[self.offsets['domain']:self.offsets['domain']+self.n*self.words]
        data[self.offsets['board']:self.offsets['board']+self.n]=0xffffffff
        data[self.offsets['owner']:self.offsets['owner']+self.n]=0xffffffff
        data[self.offsets['parent']:self.offsets['parent']+self.n*16]=np.arange(self.n*16,dtype=np.uint32)[:,None]
        data[self.offsets['size']:self.offsets['size']+self.n*16]=1
        if self.watched_clauses:
            data[self.offsets['watch_positions']:self.offsets['watch_positions']+64]=0xffffffff
            data[self.offsets['watch_pending']]=0xffffffff
        if self.global_matching:
            for name in ('matching_cells','matching_owners','matching_parent'):
                data[self.offsets[name]:self.offsets[name]+self.n]=0xffffffff
            data[self.offsets['matching_components']:self.offsets['matching_components']+self.n]=np.arange(self.n,dtype=np.uint32)[:,None]
            for word in range(self.reason_words):data[self.offsets['matching_dirty']+word]=(1<<min(32,self.n-32*word))-1
        if self.event_propagation:
            data[3]=self.n
            for word in range((self.n+31)//32):data[self.offsets['dirty_cells']+word]=(1<<min(32,self.n-32*word))-1
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
        arrays=[self.problem.neighbors.astype(np.int32).ravel().view(np.uint32),self.problem.sides.astype(np.int32).ravel().view(np.uint32),
            masks.ravel(),reverse.ravel(),np.array(self.problem._segment_counts,np.uint32),ports.ravel(),compatible.ravel()]
        if self.event_propagation:
            # Original domains are a static superset of every lane and restored
            # branch. This index skips impossible tile consumers without owning
            # or proving any extra search work.
            consumers=np.zeros((self.n,(self.n+31)//32),np.uint32)
            for cell,row in enumerate(self.problem.domains):
                for tile in {int(code)//3 for code in row}:consumers[tile,cell//32]|=np.uint32(1<<(cell%32))
            arrays.append(consumers.ravel())
        return np.concatenate(arrays)
    def step(self,work=1024):
        if type(work) is not int or not 1<=work<=1000000:raise ValueError('Work quantum must be1..1000000.')
        self.kernel(((self.lanes+3)//4,) if self.cooperative else ((self.lanes+63)//64,),(128,) if self.cooperative else (64,),(self.state,self.tables,np.int32(self.lanes),np.int32(work)))
        self.cp.cuda.get_current_stream().synchronize()
        return self.state[self.offsets['meta']].get().astype(int)
    def propagate_root(self,work=1024):
        """Advance root inference only; phase2 is a resumable fixed-point barrier."""
        if type(work)is not int or not 1<=work<=1000000:raise ValueError('Work quantum must be1..1000000.')
        if bool(self.cp.any(self.state[1]!=0)):raise ValueError('Root preparation cannot contain decisions.')
        self.propagate_kernel(((self.lanes+3)//4,) if self.cooperative else ((self.lanes+63)//64,),(128,) if self.cooperative else (64,),(self.state,self.tables,np.int32(self.lanes),np.int32(work)))
        self.cp.cuda.get_current_stream().synchronize()
        meta=self.state[:7].get()
        return meta[0].astype(int),bool(np.all((meta[0]!=0)|(meta[6]==2)))
    def fork_root(self,cubes,*,max_memory_bytes=512*1024*1024,extra_cubes=()):
        """Clone proved root facts into a complete disjoint partition of one root.

        Removal trails and reasons are rebased: each child cube now owns the
        propagated restrictions. Root assignments/unions remain below all future
        decision rollback markers. No parent counters or clauses are duplicated.
        """
        meta=self.state[:META].get()
        if self.lanes!=1 or self.delivered or int(meta[0,0])!=0 or int(meta[1,0])!=0 or int(meta[6,0])!=2:
            raise ValueError('Fork requires one undelivered root at propagation fixed point.')
        domains=self.state[self.offsets['domain']:self.offsets['domain']+self.n*self.words].get()
        parent=tuple(sum(int(domains[cell*self.words+word,0])<<(32*word) for word in range(self.words)) for cell in range(self.n))
        cubes=tuple(self._validate_cube(row) for row in cubes)
        if any(any(value&~allowed for value,allowed in zip(row,parent)) for row in cubes):raise ValueError('Fork escapes propagated parent.')
        if sum(math.prod(value.bit_count() for value in row) for row in cubes)!=math.prod(value.bit_count() for value in parent):
            raise ValueError('Fork does not cover the propagated parent.')
        extra_cubes=tuple(self._validate_cube(row) for row in extra_cubes)
        child=DomainCUDA(self.problem,(*cubes,*extra_cubes),target=self.target,max_memory_bytes=max_memory_bytes,
                         learning=self.learning,incremental_strands=self.incremental_strands,joint_support=self.joint_support,event_propagation=self.event_propagation,precise_reasons=self.precise_reasons,reason_slots=self.reason_slots if self.precise_reasons else None,support_reuse=self.support_reuse,support_metrics=self.support_metrics,support_slots=self.support_slots if self.support_reuse else None,watched_clauses=self.watched_clauses,clause_metrics=self.clause_metrics,global_matching=self.global_matching,cooperative=self.cooperative,proof_transport=self.proof_transport)
        try:
            for name,length in [('board',self.n),('owner',self.n),('assign_log',self.n),('parent',self.n*16),('size',self.n*16),
                                ('union_child',self.n*16),('union_parent',self.n*16),('union_size',self.n*16)]:
                offset=self.offsets[name];child.state[offset:offset+length,:len(cubes)]=self.state[offset:offset+length]
            if self.support_reuse:
                # Positive witnesses survive rebasing only as checked hints;
                # narrower child domains are revalidated on every hit.
                for name,length in [('support_keys',self.support_slots),('support_values',3*self.support_slots)]:
                    start=self.offsets[name];child.state[start:start+length,:len(cubes)]=self.state[start:start+length]
            child.state[9:11,:len(cubes)]=self.state[9:11]
            self.cp.cuda.get_current_stream().synchronize()
            return child
        except BaseException:
            child.close();raise
    def deduction_input(self,start=0):
        """One bounded failed-lane trace. This is untrusted proof input only."""
        if not self.proof_transport:return None
        if self.delivered:raise ValueError('Enumeration history cannot produce puzzle proofs.')
        if type(start)is not int or start<0:raise ValueError('Invalid proof lane cursor.')
        info=self.state[[0,6,self.offsets['proof_meta']+1,self.offsets['proof_meta']+2]].get()
        for offset in range(self.lanes):
            lane=(start+offset)%self.lanes
            if not (info[2,lane] or info[3,lane]) or not (info[0,lane]==2 or info[1,lane]==3):continue
            data=self.state[:,lane].get();depth=int(data[self.offsets['proof_meta']]);count=int(data[2])
            decisions=[(int(data[self.offsets['frame_cell']+level]),1<<int(data[self.offsets['frame_code']+level])) for level in range(depth)]
            trace={'root':list(self.cubes[lane]),'decisions':decisions,'trail':[int(x) for x in data[self.offsets['trail']:self.offsets['trail']+count]]}
            trace['identity']=hashlib.sha256(json.dumps([self.binding,trace],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            trace['next']=(lane+1)%self.lanes;return trace
        return None
    def learned_claims(self,start=0,*,lanes=4,limit=16):
        """Read a bounded sample of local claims; these have NO global authority."""
        if self.delivered:raise ValueError('Do not export enumeration exclusions as puzzle conflicts.')
        if type(start)is not int or start<0 or type(lanes)is not int or not 1<=lanes<=16 or type(limit)is not int or not 1<=limit<=128:
            raise ValueError('Invalid bounded conflict harvest.')
        first_lane=(start//32)%self.lanes;first_slot=start%32
        ids=[(first_lane+offset)%self.lanes for offset in range(min(lanes,self.lanes))]
        counts=self.state[8,ids].get();lengths=self.state[self.offsets['clause_length']:self.offsets['clause_length']+32,ids].get()
        cells=self.state[self.offsets['clause_cell']:self.offsets['clause_cell']+32*self.n,ids].get()
        codes=self.state[self.offsets['clause_code']:self.offsets['clause_code']+32*self.n,ids].get()
        records=[];oversize=0
        for column,count in enumerate(counts):
            for index in range(first_slot if column==0 else 0,int(count)):
                length=int(lengths[index,column])
                if not length:continue
                if length>32:oversize+=1;continue
                records.append([(int(cells[index*self.n+pos,column]),int(codes[index*self.n+pos,column])) for pos in range(length)])
                if len(records)>=limit:return records,(ids[column]*32+index+1)%(self.lanes*32),oversize
        return records,((ids[-1]+1)*32)%(self.lanes*32),oversize
    def import_verified_clauses(self,store,*,limit=8):
        """Only an original-root independently verified in-process store may supply cuts."""
        from constraint_shared_conflicts import VerifiedConflictStore
        if not isinstance(store,VerifiedConflictStore) or not store.matches(self.problem,self.target):raise ValueError('GPU clauses require a matching original-root verifier.')
        if type(limit)is not int or not 1<=limit<=8:raise ValueError('Verified GPU refresh permits1..8 clauses.')
        if not self.learning:raise ValueError('Verified GPU sharing requires learned-clause processing.')
        records=sorted(store.records(),key=lambda row:(len(row),row))[:limit]
        if any(not row for row in records):raise ValueError('Handle verified original-root contradiction at the coordinator.')
        if not records:return {'inserted':0,'comparisons':0}
        data=np.full((len(records),65),0xffffffff,np.uint32)
        for index,row in enumerate(records):
            data[index,0]=len(row)
            for pos,(cell,code) in enumerate(row):data[index,1+pos*2]=cell;data[index,2+pos*2]=code
        source=self.cp.asarray(data);counts=self.cp.zeros((2,self.lanes),np.uint32)
        self.share_kernel(((self.lanes+63)//64,),(64,),(self.state,source,counts,np.int32(self.lanes),np.int32(len(records))))
        self.cp.cuda.get_current_stream().synchronize();counts=counts.get().astype(np.uint64)
        return {'inserted':int(counts[0].sum()),'comparisons':int(counts[1].sum())}
    def _lane_ids(self,lanes):
        ids=list(range(self.lanes)) if lanes is None else list(lanes)
        if len(set(ids))!=len(ids) or any(type(lane)is not int or not 0<=lane<self.lanes for lane in ids):raise ValueError('Invalid GPU lane selection.')
        return ids
    def stats(self,lanes=None,*,snapshot=None):
        ids=self._lane_ids(lanes)
        def read(start,count):return (self.state[start:start+count,ids].get() if snapshot is None else snapshot[start:start+count,ids]).astype(np.uint64)
        m=read(0,META)
        extra=dict.fromkeys(('joint_support_checks','joint_subset_checks','joint_exclusions','joint_word_scans','joint_tile_visits'),0)
        if self.joint_support:
            counters=read(self.offsets['joint_counters'],10)
            extra={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(extra)}
        events=dict.fromkeys(('event_enqueues','event_cell_scans','event_inventory_checks','event_queue_word_scans','event_inventory_words'),0)
        if self.event_propagation:
            counters=read(self.offsets['event_counters'],10)
            events={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(events)}
        precise=dict.fromkeys(('precise_probes','precise_hits','precise_fallbacks','precise_stores','precise_invalidations','precise_support_words','precise_support_values','precise_dropped'),0)
        if self.precise_reasons:
            counters=read(self.offsets['precise_counters'],16)
            precise={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(precise)}
        from gpu_reasoning_round26 import SUPPORT_COUNTERS
        supports=dict.fromkeys(SUPPORT_COUNTERS,0)
        if self.support_metrics:
            counters=read(self.offsets['support_counters'],14)
            supports={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(supports)}
        from gpu_reasoning_round26 import CLAUSE_COUNTERS
        clauses=dict.fromkeys(CLAUSE_COUNTERS,0)
        if self.clause_metrics:
            counters=read(self.offsets['clause_counters'],10)
            clauses={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(clauses)}
        from gpu_reasoning_round26 import MATCHING_COUNTERS
        matching=dict.fromkeys(MATCHING_COUNTERS,0)
        if self.global_matching:
            counters=read(self.offsets['matching_counters'],18)
            matching={name:int((counters[2*i]+(counters[2*i+1]<<np.uint64(32))).sum()) for i,name in enumerate(matching)}
        return {**matching,**clauses,**supports,**precise,**events,**extra,'lanes':len(ids),'state_bytes':self.rows*len(ids)*4,'active':int(np.sum(m[0]==0)),
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
    def remaining_domain_cubes(self,lanes=None,*,snapshot=None):
        ids=self._lane_ids(lanes)
        data=(self.state.get() if lanes is None else self.state[:,ids].get()) if snapshot is None else (snapshot if lanes is None else snapshot[:,ids]);result=[]
        for lane in range(len(ids)):
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
    def clear_lane_counters(self,lanes):
        ids=self._lane_ids(lanes)
        if not ids:return
        rows=list(range(11,19))+list(range(20,28))
        for name,length in [('joint_counters',10),('event_counters',10),('precise_counters',16),('support_counters',14),('clause_counters',10),('matching_counters',18)]:
            if name in self.offsets:rows.extend(range(self.offsets[name],self.offsets[name]+length))
        row_ids=self.cp.asarray(rows);lane_ids=self.cp.asarray(ids)
        self.state[row_ids[:,None],lane_ids[None,:]]=0
        self.cp.cuda.get_current_stream().synchronize()
    def replace_lanes(self,replacements):
        """Rebase explicitly transferred task roots without touching other lanes.

        The coordinator proves coverage transfer. This method enforces domain
        scope/disjointness and commits both device columns before publishing new
        host roots. It never starts a search kernel during the ownership change.
        """
        if self.delivered:raise ValueError('Cannot repartition acknowledged enumeration state.')
        if not isinstance(replacements,dict) or not replacements:raise ValueError('Nonempty lane replacements required.')
        ids=self._lane_ids(replacements);roots=[self._validate_cube(replacements[lane]) for lane in ids]
        updated=list(self.cubes)
        for lane,row in zip(ids,roots):updated[lane]=row
        from constraint_work import disjoint
        if not disjoint(updated):raise ValueError('Replacement lane roots overlap.')
        # Allocation and validation occur before mutation. If the copy fails,
        # restore selected old columns when the CUDA context remains usable.
        previous=self.state[:,ids].get();incoming=self.cp.asarray(self._initial_state(roots))
        try:
            self.state[:,ids]=incoming;self.cp.cuda.get_current_stream().synchronize()
        except BaseException:
            self.state[:,ids]=self.cp.asarray(previous);self.cp.cuda.get_current_stream().synchronize();raise
        self.cubes=tuple(updated)
    def checkpoint(self):
        # Native arrays are retained exactly; portable cube transport separately
        # preserves remaining coverage while intentionally losing engine scratch.
        data=self.state.get()
        return {'schema':self.schema,'binding':self.binding,'cubes':[[hex(v) for v in cube] for cube in self.cubes],
                'state':data,'state_sha256':hashlib.sha256(memoryview(data).cast('B')).hexdigest(),'delivered':list(self.delivered),'rows':self.rows}
    def restore(self,checkpoint):
        if checkpoint.get('schema')!=self.schema or checkpoint.get('binding')!=self.binding or checkpoint.get('cubes')!=[[hex(v) for v in cube] for cube in self.cubes]:raise ValueError('Domain checkpoint scope differs.')
        state=checkpoint.get('state')
        if not isinstance(state,np.ndarray) or state.dtype!=np.uint32 or state.shape!=(self.rows,self.lanes):raise ValueError('Malformed domain checkpoint arrays.')
        if checkpoint.get('state_sha256')!=hashlib.sha256(memoryview(state).cast('B')).hexdigest():raise ValueError('Domain checkpoint state checksum differs.')
        # A checksum is integrity protection, not a mathematical proof. Local
        # native checkpoints are trusted process state; portable imports use
        # explicit cubes and reconstruct all deductions.
        if np.any(state[0]>2) or np.any(state[1]>self.n) or np.any(state[2]>self.n*self.nc) or np.any(state[9]>self.n) or np.any(state[10]>self.n*16):raise ValueError('Invalid domain checkpoint counters.')
        if np.any(state[3]>self.n) or np.any(state[4]>self.nc) or np.any(state[6]>(5 if self.global_matching else 4)) or np.any(state[7]>32) or np.any(state[8]>32) or np.any(state[19]>=32):raise ValueError('Invalid domain checkpoint scan state.')
        for name,length,maximum,sentinel in [('board',self.n,self.nc,True),('owner',self.n,self.n,True),('parent',self.n*16,self.n*16,False)]:
            values=state[self.offsets[name]:self.offsets[name]+length]
            if np.any((values>=maximum)&(values!=0xffffffff if sentinel else True)):raise ValueError('Invalid domain checkpoint '+name)
        for lane in range(self.lanes):
            trail=state[self.offsets['trail']:self.offsets['trail']+int(state[2,lane]),lane]
            if np.any(trail>=self.n*self.nc):raise ValueError('Invalid domain removal trail.')
            for level in range(int(state[1,lane])):
                if state[self.offsets['frame_cell']+level,lane]>=self.n or state[self.offsets['frame_code']+level,lane]>=self.nc:raise ValueError('Invalid domain decision frame.')
                if state[self.offsets['frame_trail']+level,lane]>state[2,lane] or state[self.offsets['frame_assign']+level,lane]>state[9,lane] or state[self.offsets['frame_union']+level,lane]>state[10,lane]:raise ValueError('Invalid domain rollback marker.')
        if self.proof_transport:
            for lane in range(self.lanes):
                depth=int(state[self.offsets['proof_meta'],lane])
                if depth>self.n:raise ValueError('Invalid proof failure depth.')
                if int(state[0,lane])==2 or int(state[6,lane])==3:
                    cells=set()
                    for level in range(depth):
                        cell=int(state[self.offsets['frame_cell']+level,lane]);code=int(state[self.offsets['frame_code']+level,lane])
                        if cell>=self.n or code>=self.nc or cell in cells or not self.cubes[lane][cell]&(1<<code):raise ValueError('Invalid proof decision prefix.')
                        cells.add(cell)
        if self.global_matching:
            from gpu_reasoning_round26 import validate_matching_state
            validate_matching_state(self,state)
        if self.watched_clauses:
            for lane in range(self.lanes):
                expected=[0]*self.n
                for slot in range(32):
                    length=int(state[self.offsets['clause_length']+slot,lane])
                    first=int(state[self.offsets['watch_positions']+2*slot,lane])
                    second=int(state[self.offsets['watch_positions']+2*slot+1,lane])
                    if first==second and first!=0xffffffff:raise ValueError('Duplicate watched literal.')
                    for index in range(2):
                        pos=int(state[self.offsets['watch_positions']+2*slot+index,lane])
                        if pos==0xffffffff:continue
                        if pos>=length or pos>=self.n:raise ValueError('Invalid watched clause literal index.')
                        cell=int(state[self.offsets['clause_cell']+slot*self.n+pos,lane])
                        if cell>=self.n:raise ValueError('Invalid watched clause cell.')
                        expected[cell]|=1<<slot
                if any(int(state[self.offsets['watch_cells']+cell,lane])!=mask for cell,mask in enumerate(expected)):
                    raise ValueError('Watched clause consumer index differs.')
                if int(state[6,lane])==2 and int(state[self.offsets['watch_pending'],lane])&((1<<int(state[8,lane]))-1):
                    raise ValueError('Claimed fixed point retains pending watched clauses.')
        if self.support_reuse:
            keys=state[self.offsets['support_keys']:self.offsets['support_keys']+self.support_slots]
            values=state[self.offsets['support_values']:self.offsets['support_values']+3*self.support_slots]
            if np.any(keys>4*self.n*self.nc) or np.any((values>=self.nc)&(values!=0xffffffff)):
                raise ValueError('Invalid support witness cache bounds.')
        # Mechanical state invariants are checked even for trusted local
        # checkpoints. A rehashed malformed rollback pointer must not launch a
        # device out-of-bounds write or an endless disjoint-set find loop.
        for lane,cube in enumerate(self.cubes):
            if getattr(self,'event_propagation',False):
                dirty=sum(int(state[self.offsets['dirty_cells']+word,lane])<<(32*word) for word in range((self.n+31)//32))
                if dirty>>self.n:raise ValueError('Dirty-cell queue contains padding bits.')
                if int(state[6,lane])==2 and (dirty or int(state[3,lane])!=self.n or int(state[4,lane])!=0 or int(state[5,lane])!=0):
                    raise ValueError('Fixed point retains pending propagation work.')
            domains=tuple(self._mask(state,lane,cell) for cell in range(self.n))
            if any(mask&~root for mask,root in zip(domains,cube)):
                raise ValueError('Domain checkpoint escapes its owned cube.')
            if self.precise_reasons:
                roots=tuple(sum(int(state[self.offsets['precise_root']+cell*self.words+word,lane])<<(32*word) for word in range(self.words)) for cell in range(self.n))
                zeros=tuple(sum(int(state[self.offsets['precise_zero']+cell*self.words+word,lane])<<(32*word) for word in range(self.words)) for cell in range(self.n))
                if roots!=cube or any(value&~root for value,root in zip(zeros,cube)):raise ValueError('Precise reason root scope differs.')
                keys=state[self.offsets['precise_keys']:self.offsets['precise_keys']+self.reason_slots,lane]
                active=[(slot,int(key)) for slot,key in enumerate(keys) if key not in (0,0xffffffff)]
                if len(active)!=int(state[self.offsets['precise_count'],lane]) or len(active)>self.reason_slots//2 or len({key for _,key in active})!=len(active):
                    raise ValueError('Invalid precise reason cache count or duplicate identity.')
                for slot,key in active:
                    if not 1<=key<=self.n*self.nc:raise ValueError('Invalid precise reason key.')
                    cell,code=divmod(key-1,self.nc)
                    if not cube[cell]&(1<<code) or domains[cell]&(1<<code):raise ValueError('Precise reason is not an active removal.')
                    start=(key*2654435761)&(self.reason_slots-1);distance=(slot-start)&(self.reason_slots-1)
                    if distance>=min(32,self.reason_slots) or any(int(keys[(start+step)&(self.reason_slots-1)])==0 for step in range(distance)):
                        raise ValueError('Unreachable precise reason hash entry.')
                    why=sum(int(state[self.offsets['precise_why']+slot*self.reason_words+word,lane])<<(32*word) for word in range(self.reason_words))
                    if why>>self.n:raise ValueError('Precise reason contains padding decision bits.')
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
    def close(self):self.state=self.tables=self.kernel=self.ack_kernel=self.propagate_kernel=self.share_kernel=self.module=None
