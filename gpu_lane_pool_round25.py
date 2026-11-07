"""Exact host ownership for bounded CUDA lane refill and remainder donation.

A lane is a temporary worker, not a coverage region. Owner roots survive every
queued/donated descendant. Native donor scratch is deliberately reconstructed.
"""
from __future__ import annotations
from constraint_work import contains,disjoint,frontier_snapshot,digest

COUNTERS=('gpu_lane_refills','gpu_lane_donations','gpu_lane_rebuilt_roots',
          'gpu_lane_discarded_frames','gpu_lane_discarded_removals',
          'gpu_lane_owners_completed','gpu_lane_donation_cubes','gpu_lane_balance_deferrals')
GAUGES=('gpu_lane_idle','gpu_lane_owner_groups','gpu_lane_queue_peak')
MAX_DONATION_QUEUE=4096

class LanePool:
    def __init__(self,scope,queued,*,counters=None):
        self.scope=scope;self.queue=list(queued);self.owners={};self.queued_owners=[];self.lanes=[];self.next_id=0
        self.counters=dict.fromkeys(COUNTERS,0)
        if counters:self.counters.update({k:v for k,v in counters.items() if k in COUNTERS})
        self.queue_peak=max(len(queued),(counters or {}).get('gpu_lane_queue_peak',0))
        for row in queued:self.queued_owners.append(self._new_owner(row))
    def _new_owner(self,row):
        owner=self.next_id;self.next_id+=1;self.owners[owner]=tuple(row);return owner
    def public_stats(self):
        return {**self.counters,'gpu_lane_idle':self.lanes.count(None),'gpu_lane_owner_groups':len(self.owners),'gpu_lane_queue_peak':self.queue_peak}
    def take_batch(self,count):
        if any(owner is not None for owner in self.lanes):raise ValueError('Cannot abandon unfinished lane ownership.')
        rows=self.queue[:count];self.lanes=self.queued_owners[:count]
        del self.queue[:count];del self.queued_owners[:count]
        return rows
    def fork(self,child_count,extra_count=0):
        if len(self.lanes)!=1 or self.lanes[0] is None:raise ValueError('Fork needs one owned parent.')
        self.lanes=[self.lanes[0]]*child_count+self.queued_owners[:extra_count]
        del self.queue[:extra_count];del self.queued_owners[:extra_count]
        self.counters['gpu_lane_refills']+=extra_count
    def retire(self,engine,states):
        ids=[lane for lane,value in enumerate(states) if value==2 and self.lanes[lane] is not None]
        if not ids:return {},0,[]
        if engine.delivered:raise ValueError('Enumeration exclusions cannot retire infeasible owners.')
        old=engine.stats(ids);engine.clear_lane_counters(ids)
        for lane in ids:self.lanes[lane]=None
        referenced=set(self.queued_owners)|set(self.lanes);finished=[]
        for owner in list(self.owners):
            if owner not in referenced:finished.append(self.owners.pop(owner))
        self.counters['gpu_lane_owners_completed']+=len(finished)
        return old,len(ids),finished
    def refill(self,engine):
        idle=[i for i,owner in enumerate(self.lanes) if owner is None]
        count=min(len(idle),len(self.queue))
        if not count:return False
        ids=idle[:count];engine.replace_lanes(dict(zip(ids,self.queue[:count])))
        for lane,owner in zip(ids,self.queued_owners):self.lanes[lane]=owner
        del self.queue[:count];del self.queued_owners[:count]
        self.counters['gpu_lane_refills']+=count;self.counters['gpu_lane_rebuilt_roots']+=count
        return True
    def donate(self,engine):
        """At most one donor; return its retired work counts, or no mutation.

        All exact remaining boxes move together. The donor and receiver take one
        box each; further boxes remain queued under their same original owner.
        """
        idle=[i for i,owner in enumerate(self.lanes) if owner is None]
        if not idle or self.queue:return None
        meta=engine.state[:3].get()
        active=[i for i,owner in enumerate(self.lanes) if owner is not None and meta[0,i]==0]
        if not active:return None
        donor=max(active,key=lambda i:(int(meta[1,i]),-i));parts=engine.remaining_domain_cubes([donor])
        if len(parts)==1:
            row=parts[0];width,axis=max((v.bit_count(),i) for i,v in enumerate(row))
            if width>1:
                mask=row[axis];part=0
                for _ in range(width//2):bit=mask&-mask;part|=bit;mask^=bit
                left=list(row);right=list(row);left[axis]=part;right[axis]=mask;parts=[tuple(left),tuple(right)]
        if len(parts)<2 or len(self.queue)+len(parts)-2>MAX_DONATION_QUEUE:
            self.counters['gpu_lane_balance_deferrals']+=1;return None
        receiver=idle[0];owner=self.lanes[donor];old=engine.stats([donor])
        # Both new device columns are committed before any ownership/queue is
        # published. No search kernel can run during this synchronous transfer.
        engine.replace_lanes({donor:parts[0],receiver:parts[1]})
        self.lanes[receiver]=owner;self.queue.extend(parts[2:]);self.queued_owners.extend([owner]*(len(parts)-2))
        self.queue_peak=max(self.queue_peak,len(self.queue))
        self.counters['gpu_lane_donations']+=1;self.counters['gpu_lane_rebuilt_roots']+=2
        self.counters['gpu_lane_discarded_frames']+=int(meta[1,donor]);self.counters['gpu_lane_discarded_removals']+=int(meta[2,donor])
        self.counters['gpu_lane_donation_cubes']+=len(parts)
        return old
    def snapshot(self):
        payload={'version':1,'owners':[[key,[hex(x) for x in row]] for key,row in sorted(self.owners.items())],
                 'lanes':list(self.lanes),'queued_owners':list(self.queued_owners),'next_id':self.next_id,'stats':self.public_stats()}
        payload['digest']=digest(payload);return payload
    @classmethod
    def restore(cls,scope,queued,engine,payload):
        if (not isinstance(payload,dict) or set(payload)!={'version','owners','lanes','queued_owners','next_id','stats','digest'}
                or type(payload['version'])is not int or payload['version']!=1 or payload['digest']!=digest({k:v for k,v in payload.items() if k!='digest'})):
            raise ValueError('Invalid GPU lane ownership checkpoint.')
        obj=cls(scope,[]);obj.queue=list(queued)
        if type(payload['next_id'])is not int or not 0<=payload['next_id']<2**63:raise ValueError('Invalid next owner ID.')
        obj.next_id=payload['next_id']
        if not isinstance(payload['owners'],list):raise ValueError('Invalid owner roots.')
        for entry in payload['owners']:
            if not isinstance(entry,list) or len(entry)!=2:raise ValueError('Invalid owner entry.')
            key,row=entry
            if type(key)is not int or not 0<=key<obj.next_id or key in obj.owners:raise ValueError('Invalid owner ID.')
            try:row=tuple(int(x,16) for x in row)
            except (ValueError,TypeError):raise ValueError('Invalid owner cube.') from None
            frontier_snapshot(scope,[row]);obj.owners[key]=row
        if not disjoint(obj.owners.values()):raise ValueError('Owner roots overlap.')
        lanes=payload['lanes'];qowners=payload['queued_owners']
        if not isinstance(lanes,list) or len(lanes)!=(engine.lanes if engine else 0) or not isinstance(qowners,list) or len(qowners)!=len(queued):
            raise ValueError('Owner task dimensions differ.')
        if any(type(key)is not int or key not in obj.owners for key in qowners):raise ValueError('Unknown queued owner.')
        if any(key is not None and (type(key)is not int or key not in obj.owners) for key in lanes):raise ValueError('Unknown lane owner.')
        if set(qowners)|{key for key in lanes if key is not None}!=set(obj.owners):raise ValueError('Unreferenced owner root.')
        for row,key in zip(queued,qowners):
            if not contains(obj.owners[key],row):raise ValueError('Queued task escapes owner.')
        if engine:
            meta=engine.state[:3].get()
            for lane,key in enumerate(lanes):
                if key is None:
                    counters=engine.stats([lane])
                    if int(meta[0,lane])!=2 or any(v for k,v in counters.items() if k not in ('lanes','state_bytes','active','pending','exhausted')):
                        raise ValueError('Retired GPU lane is not inactive and accounted.')
                elif not contains(obj.owners[key],engine.cubes[lane]):raise ValueError('Native lane escapes owner.')
        stats=payload['stats']
        if (not isinstance(stats,dict) or set(stats)!=set(COUNTERS+GAUGES)
                or any(type(v)is not int or v<0 for v in stats.values())
                or stats['gpu_lane_idle']!=lanes.count(None) or stats['gpu_lane_owner_groups']!=len(obj.owners)
                or stats['gpu_lane_queue_peak']<len(queued)):raise ValueError('Invalid GPU ownership statistics.')
        obj.lanes=list(lanes);obj.queued_owners=list(qowners);obj.counters={k:stats[k] for k in COUNTERS};obj.queue_peak=stats['gpu_lane_queue_peak']
        return obj
