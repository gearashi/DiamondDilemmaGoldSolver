"""Explicit trusted-local Python seam/inventory and strong-stage continuation.

Only atomic, sound domain reductions commit. Native or opaque strong callbacks
may retry their own interrupted call; completed earlier stages and the Python
basic queue are preserved. This state is not an external proof certificate.
"""
from collections import deque
import copy

STAT_KEYS=('propagation_cursor_builds','propagation_cursor_resumes','propagation_cursor_stage_skips',
 'propagation_cursor_basic_steps','propagation_cursor_evictions','propagation_cursor_opaque_retries')

class BasicCursor:
    def __init__(self,domains,arcs,incoming,tile_bits,starting=None):
        self.domains=list(domains);self.initial=tuple(domains);self.arcs=tuple(tuple(row) for row in arcs);self.incoming=tuple(tuple(row) for row in incoming);self.tile_bits=tuple(tile_bits)
        n=len(domains)
        if not n or len(incoming)!=n or len(tile_bits)!=n or any(type(m)is not int or m<0 or m.bit_length()>3*n for m in domains):raise ValueError('Invalid basic propagation dimensions.')
        if any(len(row)!=4 or any(type(v)is not int for v in row) or not 0<=row[0]<n or not 0<=row[2]<n or row[1] not in (0,1,2) or row[3] not in (0,1,2) for row in self.arcs):raise ValueError('Invalid propagation arcs.')
        if any(type(index)is not int or not 0<=index<len(arcs) or self.arcs[index][2]!=cell for cell,row in enumerate(self.incoming) for index in row):raise ValueError('Invalid incoming propagation index.')
        if any(type(mask)is not int or mask!=7<<(3*tile) for tile,mask in enumerate(tile_bits)):raise ValueError('Invalid physical tile masks.')
        self.projections={};self.native_pending=False;self.reset(starting)
    def reset(self,starting=None):
        indices=range(len(self.arcs)) if starting is None else starting
        self.queue=deque(dict.fromkeys(indices));self.pending=set(self.queue)
        if any(type(i)is not int or not 0<=i<len(self.arcs) for i in self.queue):raise ValueError('Invalid initial arc queue.')
        self.phase='arcs';self.before=tuple(self.domains);self.owners={};self.occurrences=[[] for _ in self.domains]
        self.scan=0;self.owner_index=0;self.occurrence_index=0;self.sole_index=0;self.failed=None;self.done=False
    def _enqueue(self,cell):
        for index in self.incoming[cell]:
            if index not in self.pending:self.queue.append(index);self.pending.add(index)
    def _narrow(self,cell,mask,reason,rule,on_narrow):
        if mask==self.domains[cell]:return False
        if mask&~self.domains[cell]:raise ValueError('Propagation may only narrow.')
        self.domains[cell]=mask;self._enqueue(cell);on_narrow(cell,mask,reason,rule);return True
    def advance(self,filter_arc,check=lambda:None,on_narrow=lambda *args:None,stats=None,native_basic=None):
        stats={} if stats is None else stats
        for key in ('python_arcs_checked','arc_revisions','tile_prunes','propagation_cursor_basic_steps'):stats.setdefault(key,0)
        while not self.done:
            check();stats['propagation_cursor_basic_steps']+=1
            if not all(self.domains):self.failed=tuple(c for c,m in enumerate(self.domains) if not m);self.done=True;break
            if self.phase=='arcs':
                if self.queue and native_basic is not None:
                    stats['propagation_cursor_opaque_retries']=stats.get('propagation_cursor_opaque_retries',0)+int(self.native_pending)
                    self.native_pending=True
                    try:possible=native_basic(self.domains,tuple(self.queue))
                    except BaseException:
                        # An opaque native call may have committed reductions
                        # before interruption without exposing its inner queue.
                        # Rebuild only that call's arc frontier conservatively.
                        self.queue=deque(range(len(self.arcs)));self.pending=set(self.queue);raise
                    self.native_pending=False;self.queue.clear();self.pending.clear()
                    if possible is False:self.failed=tuple(range(len(self.domains)));self.done=True;break
                if self.queue:
                    index=self.queue[0];a,sa,b,sb=self.arcs[index]
                    keep=filter_arc(self.domains[a],self.domains[b],sa,sb)
                    if type(keep)is not int or keep<0 or keep&~self.domains[a]:raise ValueError('Arc filter widened domains.')
                    self.queue.popleft();self.pending.remove(index);stats['python_arcs_checked']+=1
                    if keep!=self.domains[a]:stats['arc_revisions']+=1;self._narrow(a,keep,(b,),'seam support',on_narrow)
                    continue
                self.before=tuple(self.domains);self.phase='scan';self.scan=0;self.owners={};self.occurrences=[[] for _ in self.domains]
            elif self.phase=='scan':
                if self.scan==len(self.domains):self.phase='owners';self.owner_index=0;self.occurrence_index=0;continue
                cell=self.scan;mask=self.domains[cell];possible=self.projections.get(mask)
                if possible is None:possible=tuple(sorted({code//3 for code in range(mask.bit_length()) if mask>>code&1}));self.projections[mask]=possible
                if len(possible)==1:
                    tile=possible[0]
                    if tile in self.owners:self.failed=(cell,self.owners[tile]);self.done=True;break
                    self.owners[tile]=cell
                for tile in possible:self.occurrences[tile].append(cell)
                self.scan+=1
            elif self.phase=='owners':
                owners=tuple(self.owners.items())
                if self.owner_index==len(owners):self.phase='sole';self.sole_index=0;continue
                tile,owner=owners[self.owner_index];cells=self.occurrences[tile]
                if self.occurrence_index==len(cells):self.owner_index+=1;self.occurrence_index=0;continue
                cell=cells[self.occurrence_index];self.occurrence_index+=1
                if cell!=owner and self.domains[cell]&self.tile_bits[tile]:
                    stats['tile_prunes']+=1;self._narrow(cell,self.domains[cell]&~self.tile_bits[tile],(owner,),'tile ownership',on_narrow)
            elif self.phase=='sole':
                if self.sole_index==len(self.domains):
                    if self.queue or tuple(self.domains)!=self.before:self.phase='arcs'
                    else:self.done=True
                    continue
                tile=self.sole_index;self.sole_index+=1;cells=self.occurrences[tile]
                if not cells:self.failed=tuple(range(len(self.domains)));self.done=True;break
                if len(cells)==1:
                    cell=cells[0];keep=self.domains[cell]&self.tile_bits[tile]
                    if keep!=self.domains[cell]:stats['tile_prunes']+=1;self._narrow(cell,keep,tuple(range(len(self.domains))),'sole tile host',on_narrow)
            else:raise ValueError('Invalid basic propagation phase.')
        return self.failed is None
    def snapshot(self):
        payload=copy.deepcopy({key:getattr(self,key) for key in ('initial','domains','phase','before','occurrences','scan','owner_index','occurrence_index','sole_index','failed','done','native_pending')})
        payload.update(queue=list(self.queue),pending=sorted(self.pending),owners=list(self.owners.items()),projections=list(self.projections.items()))
        return payload
    @classmethod
    def restore(cls,payload,arcs,incoming,tile_bits):
        if type(payload)is not dict:raise ValueError('Invalid basic cursor.')
        fields={'initial','domains','queue','pending','phase','before','owners','occurrences','scan','owner_index','occurrence_index','sole_index','failed','done','projections','native_pending'}
        if set(payload)!=fields:raise ValueError('Basic cursor schema differs.')
        obj=cls(payload['initial'],arcs,incoming,tile_bits);n=len(obj.domains)
        def masks(row):return isinstance(row,(tuple,list)) and len(row)==n and all(type(m)is int and m>=0 and not m&~old for m,old in zip(row,obj.initial))
        if not masks(payload['domains']) or not masks(payload['before']) or payload['phase'] not in ('arcs','scan','owners','sole') or type(payload['done'])is not bool or type(payload['native_pending'])is not bool:raise ValueError('Invalid basic cursor domains/phase.')
        for key in ('scan','owner_index','occurrence_index','sole_index'):
            if type(payload[key])is not int or not 0<=payload[key]<=n:raise ValueError('Invalid basic cursor position.')
        if not isinstance(payload['queue'],(tuple,list)) or not isinstance(payload['pending'],(tuple,list)) or any(type(i)is not int for i in (*payload['queue'],*payload['pending'])):raise ValueError('Malformed queue.')
        queue=list(payload['queue']);pending=set(payload['pending'])
        if len(queue)!=len(set(queue)) or set(queue)!=pending or any(type(i)is not int or not 0<=i<len(arcs) for i in queue):raise ValueError('Invalid saved arc queue.')
        owners=payload['owners']
        if not isinstance(owners,(tuple,list)) or any(not isinstance(row,(tuple,list)) or len(row)!=2 or any(type(v)is not int or not 0<=v<n for v in row) for row in owners) or len({row[0] for row in owners})!=len(owners):raise ValueError('Invalid saved tile ownership.')
        if len(payload['occurrences'])!=n or any(not isinstance(row,(tuple,list)) or len(row)!=len(set(row)) or any(type(c)is not int or not 0<=c<n for c in row) for row in payload['occurrences']):raise ValueError('Invalid saved tile occurrences.')
        if payload['failed'] is not None and (not isinstance(payload['failed'],(tuple,list)) or any(type(c)is not int or not 0<=c<n for c in payload['failed'])):raise ValueError('Invalid saved failure.')
        projections=payload['projections']
        if not isinstance(projections,(tuple,list)) or any(not isinstance(row,(tuple,list)) or len(row)!=2 for row in projections):raise ValueError('Invalid saved tile projection.')
        if any(type(m)is not int or m<0 or m.bit_length()>3*n or not isinstance(v,(tuple,list)) or any(type(t)is not int for t in v) or tuple(v)!=tuple(sorted({c//3 for c in range(m.bit_length()) if m>>c&1})) for m,v in projections) or len({row[0] for row in projections})!=len(projections):raise ValueError('Invalid saved tile projection.')
        for key,value in payload.items():setattr(obj,key,copy.deepcopy(value))
        obj.initial=tuple(obj.initial);obj.before=tuple(obj.before);obj.domains=list(obj.domains);obj.queue=deque(queue);obj.pending=pending;obj.owners=dict(owners);obj.projections={m:tuple(v) for m,v in projections}
        if obj.owner_index>len(obj.owners) or obj.phase=='scan' and any(c>=obj.scan for row in obj.occurrences for c in row):raise ValueError('Inventory cursor skipped unfinished cells.')
        if obj.phase=='owners' and obj.owner_index<len(obj.owners) and obj.occurrence_index>len(obj.occurrences[tuple(obj.owners)[obj.owner_index]]):raise ValueError('Inventory cursor exceeds occurrences.')
        return obj


class PropagationCursor:
    def __init__(self,domains,arcs,incoming,tile_bits,stage_names,starting=None):
        self.basic=BasicCursor(domains,arcs,incoming,tile_bits,starting);self.names=tuple(stage_names)
        if any(type(name)is not str for name in self.names) or len(set(self.names))!=len(self.names):raise ValueError('Propagation stages need distinct names.')
        self.phase='basic';self.stage=0;self.before=None;self.done=False;self.possible=None;self.opaque_pending=False;self.stats=dict.fromkeys(STAT_KEYS,0)
    @property
    def domains(self):return self.basic.domains
    def attach(self,domains):
        if type(domains)is not list or domains!=self.domains:raise ValueError('Propagation attachment differs from retained domains.')
        self.basic.domains=domains;return self
    def advance(self,filter_arc,actions,check=lambda:None,on_narrow=lambda *args:None,stats=None,native_basic=None):
        if tuple(name for name,_ in actions)!=self.names:raise ValueError('Propagation stage policy differs.')
        while not self.done:
            check()
            if self.phase=='basic':
                if not self.basic.advance(filter_arc,check,on_narrow,stats,native_basic):self.done=True;self.possible=False;break
                self.phase='strong';self.stage=0;self.before=tuple(self.domains)
            elif self.stage<len(actions):
                self.stats['propagation_cursor_opaque_retries']+=int(self.opaque_pending);self.opaque_pending=True
                result=actions[self.stage][1](self.domains)
                self.opaque_pending=False;self.stage+=1
                if result is False or not all(self.domains):self.done=True;self.possible=False;break
            else:
                if tuple(self.domains)==self.before:self.done=True;self.possible=True;break
                changed=[c for c,(a,b) in enumerate(zip(self.before,self.domains)) if a!=b]
                self.basic.reset(index for cell in changed for index in self.basic.incoming[cell]);self.phase='basic'
        return self.possible
    def snapshot(self):return {'basic':self.basic.snapshot(),'names':self.names,'phase':self.phase,'stage':self.stage,'before':self.before,'done':self.done,'possible':self.possible,'opaque_pending':self.opaque_pending,'stats':dict(self.stats)}
    @classmethod
    def restore(cls,payload,arcs,incoming,tile_bits,stage_names):
        keys={'basic','names','phase','stage','before','done','possible','opaque_pending','stats'}
        if type(payload)is not dict or set(payload)!=keys or tuple(payload['names'])!=tuple(stage_names):raise ValueError('Propagation cursor schema/policy differs.')
        basic=BasicCursor.restore(payload['basic'],arcs,incoming,tile_bits);obj=cls(basic.initial,arcs,incoming,tile_bits,stage_names);obj.basic=basic
        if payload['phase'] not in ('basic','strong') or type(payload['stage'])is not int or not 0<=payload['stage']<=len(stage_names) or type(payload['done'])is not bool or type(payload['opaque_pending'])is not bool or payload['possible'] is not None and type(payload['possible'])is not bool:raise ValueError('Invalid stage position/status.')
        if payload['phase']=='strong' and (not isinstance(payload['before'],(list,tuple)) or len(payload['before'])!=len(basic.domains) or any(type(m)is not int or m<0 or m&~root for m,root in zip(payload['before'],basic.initial))):raise ValueError('Missing strong-phase input.')
        if payload['done']!=(payload['possible'] is not None) or type(payload['stats'])is not dict or set(payload['stats'])!=set(STAT_KEYS) or any(type(v)is not int or v<0 for v in payload['stats'].values()):raise ValueError('Invalid phase completion/counters.')
        for key in keys-{'basic','names'}:setattr(obj,key,copy.deepcopy(payload[key]))
        obj.before=None if obj.before is None else tuple(obj.before)
        return obj
