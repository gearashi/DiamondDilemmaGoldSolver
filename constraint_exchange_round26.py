"""Bounded cross-engine claims, independent verification, then publication.

Imported bytes and worker identity are not proof authority. Only original-root
verified records can retire unleased symbolic work. Local-root records are never
silently widened; they remain local unless their producer supplies checked guards.
"""
from constraint_knowledge import KnowledgeStore
from constraint_shared_conflicts import VerifiedConflictStore
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_work import digest

class CheckedExchange:
    def __init__(self,problem,target,options):
        self.problem=problem;self.target=target;self.options=options
        self.knowledge=KnowledgeStore(problem,target=target) if target=='gold' else None
        self.shared=VerifiedConflictStore(problem,target) if options['shared_conflicts'] or options['gpu_shared_clauses'] else None
        self.masks=VerifiedMaskConflictStore(problem,target,proof_transport=options['proof_transport'],
            resumable_proof_checking=options['resumable_proof_checking'],mixed_proof_dag=options['mixed_proof_dag'],
            proof_indexing=options['proof_indexing']) if options['smart_learning'] else None
        if self.masks is not None:self.masks.configure_isolated_producer(options['isolated_proof_producer'])
        self.stats={'exchange_payloads':0,'exchange_rejected_payloads':0,'exchange_verified_placements':0,
                    'exchange_verified_masks':0,'exchange_verified_loops':0,'exchange_verification_calls':0}
        self.last=None

    def offer(self,payload):
        if not isinstance(payload,dict):raise ValueError('Malformed exchange container.')
        for key,store in (('knowledge',self.knowledge),('shared_conflicts',self.shared),('mask_conflicts',self.masks)):
            item=payload.get(key)
            if item is None or store is None:continue
            self.stats['exchange_payloads']+=1
            try:
                if key=='knowledge':store.merge_snapshot(item)
                else:store.import_snapshot(item)
            except (ValueError,TypeError,KeyError):self.stats['exchange_rejected_payloads']+=1

    def advance(self,check=lambda:None,seconds=.005):
        for store in (self.shared,self.masks):
            if store is not None:store.advance(check,seconds=seconds,attempts=1)
        self.stats['exchange_verified_placements']=len(self.shared.records()) if self.shared is not None else 0
        self.stats['exchange_verified_masks']=len(self.masks.records()) if self.masks is not None else 0
        self.stats['exchange_verified_loops']=len(self.knowledge.records()) if self.knowledge is not None else 0
        self.stats['exchange_verification_calls']=(self.shared.stats['shared_conflicts_calls'] if self.shared is not None else 0)+(self.masks.stats['mask_conflicts_calls'] if self.masks is not None else 0)

    def snapshot(self,*,pending=False):
        return {**({'knowledge':self.knowledge.snapshot()} if self.knowledge is not None else {}),
                **({'shared_conflicts':self.shared.snapshot()} if self.shared is not None else {}),
                **({'mask_conflicts':self.masks.snapshot(include_pending=pending)} if self.masks is not None else {})}

    def retire_verified(self,frontier):
        records=[]
        for owner in (self.knowledge,self.shared):
            if owner is not None:records.extend(tuple((cell,1<<code) for cell,code in row) for row in owner.records())
        if self.masks is not None:records.extend(self.masks.records())
        signature=digest(records)
        # Reclaimed/migrated leases can expose fresh pending assignments to an
        # unchanged proof set. Canonical IDs belong to one specific node pool.
        identity=(id(frontier),id(frontier.sets.graph),signature,frontier.pending)
        if identity==self.last:return
        boxes=[]
        for record in records:
            box=list(frontier.scope.root)
            for cell,mask in record:box[cell]&=mask
            if all(box):boxes.append(tuple(box))
        frontier.exclude_verified(boxes);self.last=(id(frontier),id(frontier.sets.graph),signature,frontier.pending)

    def close(self):
        for store in (self.shared,self.masks):
            if store is not None:store.close()


def import_for_worker(state,problem,target,options,payload):
    if not payload:return
    if target=='gold' and payload.get('knowledge') is not None:
        owner=state.get('_knowledge')
        if owner is None:owner=state['_knowledge']=KnowledgeStore(problem,target=target)
        owner.merge_snapshot(payload['knowledge'])
    if payload.get('shared_conflicts') is not None and (options['shared_conflicts'] or options['gpu_shared_clauses']):
        owner=state.get('_shared_conflicts')
        if owner is None:owner=state['_shared_conflicts']=VerifiedConflictStore(problem,target)
        owner.import_snapshot(payload['shared_conflicts'])
    if payload.get('mask_conflicts') is not None and options['smart_learning']:
        owner=state.get('_mask_conflicts')
        if owner is None:
            owner=state['_mask_conflicts']=VerifiedMaskConflictStore(problem,target,proof_transport=options['proof_transport'],
                resumable_proof_checking=options['resumable_proof_checking'],mixed_proof_dag=options['mixed_proof_dag'],proof_indexing=options['proof_indexing'])
        owner.configure_isolated_producer(options['isolated_proof_producer'])
        owner.import_snapshot(payload['mask_conflicts'])
