"""Consolidated production integration for regional workstreams F09-F13.

All controls are opt-in. Work caps bound individual reasoning queries, never the
owning search region. Only completed support/proof results mutate domains;
positive full-board witnesses are independently validated again by DFS.
"""
from constraint_regions import HierarchicalRegionPruner
from constraint_region_compiler_round26 import RegionalCompiler,STAT_KEYS as COMPILER_KEYS
from constraint_region_interfaces_round26 import EnvelopeCache,OverlapConsistency,STAT_KEYS as ENVELOPE_KEYS
from constraint_region_tree_round26 import RecursiveRegions,STAT_KEYS as TREE_KEYS
from constraint_region_proof_round26 import RegionalProofReuse,STAT_KEYS as PROOF_KEYS
from constraint_connectivity_round26 import ConnectivityObligations,STAT_KEYS as CONNECTIVITY_KEYS
from constraint_region_integer_round26 import IntegerRegionalMaster,STAT_KEYS as INTEGER_KEYS
from constraint_separator import _Budget
from constraint_region_incremental import source_binding
from constraint_regional_proof_bridge_round26 import RegionalProofQueue,STAT_KEYS as CHAIN_KEYS,QUEUE_KEYS

KEYS=('regional_compiler','recursive_regions','regional_proof_reuse','connectivity_obligations','integer_regional_master')
EXTRA_KEYS=('regional_overlap_queries','regional_overlap_cache_hits','regional_overlap_state_steps','regional_overlap_prunes')
STAT_KEYS=tuple(dict.fromkeys((*COMPILER_KEYS,*ENVELOPE_KEYS,*TREE_KEYS,*PROOF_KEYS,*CONNECTIVITY_KEYS,*INTEGER_KEYS,*EXTRA_KEYS,*CHAIN_KEYS,*QUEUE_KEYS)))
GAUGE_KEYS=()

class RegionalWorkstreams:
    KEYS=KEYS
    def __init__(self,problem,target,options):
        self.problem=problem;self.target=target;self.options=dict(options)
        self.owner=HierarchicalRegionPruner(problem,target);self.semantic=source_binding(self.owner)
        self.compiler=RegionalCompiler(self.owner);self.envelopes=EnvelopeCache(self.owner);self.overlap=OverlapConsistency(self.owner)
        self.tree=RecursiveRegions(self.owner);self.proofs=RegionalProofReuse(self.owner)
        self.connectivity=ConnectivityObligations(self.owner);self.integer=IntegerRegionalMaster(self.owner)
        self.solution=None;self._extra=dict.fromkeys(EXTRA_KEYS,0)
        self.proof_queue=RegionalProofQueue(problem,target) if options.get('smart_learning') else None
    @property
    def stats(self):
        result=dict.fromkeys(STAT_KEYS,0)
        for stage in (self.compiler,self.envelopes,self.tree,self.proofs,self.connectivity,self.integer):
            for name,value in stage.stats.items():result[name]+=value
        result.update(self._extra)
        if self.proof_queue is not None:result.update(self.proof_queue.stats)
        return result
    def configure_cache_dir(self,path):
        for stage in (self.compiler,self.tree,self.proofs,self.integer):stage.configure_cache_dir(path)
    def _local_relations(self,domains,check):
        # A connected envelope is chosen from the least ambiguous active cells.
        # Its exact interface remains reusable when outside domains or chains vary.
        eligible={c for c,d in enumerate(domains) if 1<d.bit_count()<=12}
        if not eligible:return False
        first=min(eligible,key=lambda c:(domains[c].bit_count(),c));cells=[first]
        while len(cells)<6:
            candidates=set().union(*(self.owner.neighbors[c] for c in cells))&eligible-set(cells)
            if not candidates:break
            cells.append(min(candidates,key=lambda c:(domains[c].bit_count(),c)))
        left=tuple(sorted(cells));view=self.envelopes.get(left,domains,check,lease=256)
        if view is None:return False
        updated=list(domains)
        for c,mask in zip(left,view['supports']):updated[c]&=mask
        # An overlapping shifted bag carries exact shared-cell assignments.
        neighbors=set().union(*(self.owner.neighbors[c] for c in left))&eligible-set(left)
        if len(left)>2 and neighbors:
            right=tuple(sorted((*left[1:],min(neighbors))))
            other=self.envelopes.get(right,domains,check,lease=256)
            if other is not None:
                before=(self.overlap.checks,self.overlap.hits,self.overlap.state_steps)
                try:
                    count,members,supports=self.overlap.supports(view,other,domains,check,max_steps=50000)
                    for c,mask in zip(members,supports):updated[c]&=mask
                except _Budget:pass
                finally:
                    self._extra['regional_overlap_queries']+=self.overlap.checks-before[0]
                    self._extra['regional_overlap_cache_hits']+=self.overlap.hits-before[1]
                    self._extra['regional_overlap_state_steps']+=self.overlap.state_steps-before[2]
        check();removed=sum((a&~b).bit_count() for a,b in zip(domains,updated))
        self._extra['regional_overlap_prunes']+=removed
        changed=updated!=list(domains);domains[:]=updated;return changed
    def _stage(self,origin,domains,action):
        before=tuple(domains);changed=action()
        if self.proof_queue is not None and tuple(domains)!=before:
            self.proof_queue.capture(origin,before,domains,
                cores=self.proofs.export_certificates() if origin=='core' else (),
                connectivity=self.connectivity.reasons if origin=='connectivity' else ())
        return changed
    def prune(self,domains,check=lambda:None):
        if source_binding(self.owner)!=self.semantic:raise ValueError('Regional workstream source changed.')
        self.solution=None;changed=False
        if not all(domains):return False
        if self.options.get('connectivity_obligations'):
            changed|=self._stage('connectivity',domains,lambda:self.connectivity.prune(domains,check))
            if not all(domains):return changed
        if self.options.get('regional_proof_reuse'):
            changed|=self._stage('core',domains,lambda:self.proofs.prune(domains,check))
            if not all(domains):return changed
        if self.options.get('regional_compiler'):
            changed|=self._stage('overlap',domains,lambda:self._local_relations(domains,check))
            if not all(domains):return changed
            changed|=self._stage('compiler',domains,lambda:self.compiler.prune(domains,check,lease=512,existence_only=self.target=='gold'))
            if self.compiler.solution is not None:self.solution=self.compiler.solution;return changed
            if not all(domains):return changed
        if self.options.get('recursive_regions'):
            changed|=self._stage('recursive',domains,lambda:self.tree.prune(domains,check,lease=1024))
            if self.tree.solution is not None:self.solution=self.tree.solution;return changed
            if not all(domains):return changed
        if self.options.get('integer_regional_master'):
            changed|=self._stage('integer',domains,lambda:self.integer.prune(domains,check,lease=1024))
            if self.integer.solution is not None:self.solution=self.integer.solution
        return changed
    def drain_deduction_chains(self,check=lambda:None,*,limit=8,max_states=2048,max_steps=20000):
        if self.proof_queue is not None:yield from self.proof_queue.drain(check,limit=limit,max_states=max_states,max_steps=max_steps)
    def exported_certificates(self):return self.proofs.export_certificates()
    def import_certificate(self,record,check=lambda:None):return self.proofs.import_certificate(record,check)
