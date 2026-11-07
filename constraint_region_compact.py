"""Whole-constraint compact seam/resource factors (round25 #16).

Direct labelled transitions replace Cartesian pair tables. Used resources leave
state only after their last possible occurrence, when they must have been used.
Every cell participates. Incomplete/capped graphs never provide support masks.
"""
from collections import OrderedDict
from constraint_pruning import values
from constraint_region_incremental import source_binding
from validator import reverse_mask

STAT_KEYS=('compact_factor_queries','compact_factor_completed','compact_factor_unknown',
 'compact_factor_admission_skips','compact_factor_cache_hits','compact_factor_placements',
 'compact_factor_transitions','compact_factor_states','compact_factor_merges',
 'compact_factor_seam_tests','compact_factor_forget_checks','compact_factor_backward_edges',
 'compact_factor_prunes','compact_factor_frontier_width','compact_factor_live_resources',
 'compact_gold_queries','compact_gold_completed','compact_gold_unknown','compact_gold_pair_splices',
 'compact_gold_premature_cycles','compact_gold_final_rejections','compact_gold_port_peak','compact_gold_validations')
GAUGE_KEYS=('compact_factor_frontier_width','compact_factor_live_resources','compact_gold_port_peak')


class CompactLimit(Exception):pass


class CompactFactors:
    def __init__(self,problem,domains,check=lambda:None,*,stats=None,max_placements=4096,
                 max_frontier=96,max_resources=32):
        self.problem=problem;self.domains=tuple(domains);self.cells=tuple(range(problem.n));self.root_state=(0,())
        self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        for bound in (max_placements,max_frontier,max_resources):
            if type(bound)is not int or bound<0:raise ValueError('Invalid compact factor bound.')
        if (len(domains)!=problem.n or any(type(mask)is not int or mask<0 or
                mask&~sum(1<<code for code in set(problem.domains[cell])) for cell,mask in enumerate(domains))):
            raise ValueError('Compact factor domains escape the original puzzle.')
        if sum(mask.bit_count() for mask in domains)>max_placements:raise CompactLimit('placement admission cap')
        self.choices=tuple(tuple(values(mask)) for mask in domains);self.n=problem.n
        self.missing=not all(domains);first={};last={}
        for cell,codes in enumerate(self.choices):
            check()
            for code in codes:
                tile=code//3;first.setdefault(tile,cell);last[tile]=cell
        self.missing|=len(last)!=self.n
        self.expires=tuple(sum(1<<tile for tile,ending in last.items() if ending==cell) for cell in self.cells)
        self.future=tuple(sum(1<<tile for tile,ending in last.items() if ending>=level) for level in range(self.n+1))
        resource_width=max((sum(start<=level<last[tile] for tile,start in first.items()) for level in self.cells),default=0)
        self.stats['compact_factor_live_resources']=max(self.stats['compact_factor_live_resources'],resource_width)
        if resource_width>max_resources:raise CompactLimit('live resource width cap')
        incident=[[] for _ in self.cells];incident_bits=[0]*self.n
        for edge,(a,sa,b,sb) in enumerate(problem.edges):
            check();incident[a].append((edge,sa,False));incident[b].append((edge,sb,True))
            if a!=b:incident_bits[a]|=1<<edge;incident_bits[b]|=1<<edge
        self.frontiers=[()];cut=0
        for cell in self.cells:
            cut^=incident_bits[cell];self.frontiers.append(tuple(values(cut)))
        width=max(map(len,self.frontiers),default=0)
        self.stats['compact_factor_frontier_width']=max(self.stats['compact_factor_frontier_width'],width)
        if width>max_frontier:raise CompactLimit('seam frontier width cap')
        self.patterns=[];self.incoming=[];self.outputs=[]
        for cell,codes in enumerate(self.choices):
            check();before={edge:index for index,edge in enumerate(self.frontiers[cell])}
            self.incoming.append(tuple((edge,before[edge]) for edge,side,rev in incident[cell] if edge in before))
            self.outputs.append(tuple((edge,before.get(edge,-1)) for edge in self.frontiers[cell+1]))
            patterns={}
            for code in codes:
                check();row={};valid=True;self.stats['compact_factor_placements']+=1
                for edge,side,rev in incident[cell]:
                    mask=int(problem.masks[code,side]);mask=reverse_mask(mask) if rev else mask
                    if edge in row and row[edge]!=mask:valid=False
                    row[edge]=mask
                patterns[code]=row if valid else None
            self.patterns.append(patterns)
    def transition(self,level,state,code,check=lambda:None):
        used,seams=state;tile=1<<(code//3)
        if used&tile:return None
        pattern=self.patterns[level][code]
        if pattern is None:return None
        for edge,index in self.incoming[level]:
            self.stats['compact_factor_seam_tests']+=1
            if pattern[edge]!=seams[index]:return None
        present=used|tile;self.stats['compact_factor_forget_checks']+=1
        if self.expires[level]&~present:return None
        following=tuple(seams[index] if index>=0 else pattern[edge] for edge,index in self.outputs[level])
        return present&self.future[level+1],following


class GoldCompactFactors(CompactFactors):
    """Full Gold factor state: exact pairing of every exposed strand end."""
    def __init__(self,problem,domains,check=lambda:None,*,max_ports=128,**kwargs):
        if type(max_ports)is not int or max_ports<0:raise ValueError('Invalid exposed Gold endpoint bound.')
        super().__init__(problem,domains,check,**kwargs);self.max_ports=max_ports;self.root_state=(0,(),())
        self.segments=tuple(tuple(((a[0]+rotation)%3*11+a[1]-1,(b[0]+rotation)%3*11+b[1]-1)
                                 for a,b in tile['segments']) for tile in problem.tiles for rotation in range(3))
        if any(not row for row in self.segments):raise ValueError('Gold compilation requires a strand on every piece.')
        self.closing=tuple(tuple((eid,a,sa,b,sb) for eid,(a,sa,b,sb) in enumerate(problem.edges)
                                 if max(a,b)==cell) for cell in self.cells)
    def transition(self,level,state,code,check=lambda:None):
        ordinary=super().transition(level,state[:2],code,check)
        if ordinary is None:return None
        mates={a:b for pair in state[2] for a,b in (pair,pair[::-1])}
        for a,b in self.segments[code]:
            a+=33*level;b+=33*level;mates[a]=b;mates[b]=a
        cycles=0
        for edge,a,sa,b,sb in self.closing[level]:
            for position in values(self.patterns[level][code][edge]):
                check();left=33*a+11*sa+position;right=33*b+11*sb+10-position
                if left==right:raise CompactLimit('degenerate self-port seam is unsupported')
                x=mates.pop(left);y=mates.pop(right);self.stats['compact_gold_pair_splices']+=1
                if x==right:cycles+=1
                else:mates[x]=y;mates[y]=x
        if cycles:
            if level!=self.n-1:
                self.stats['compact_gold_premature_cycles']+=1;return None
            if cycles!=1 or mates:
                self.stats['compact_gold_final_rejections']+=1;return None
        elif level==self.n-1:
            self.stats['compact_gold_final_rejections']+=1;return None
        self.stats['compact_gold_port_peak']=max(self.stats['compact_gold_port_peak'],len(mates))
        if len(mates)>self.max_ports:raise CompactLimit('exposed Gold endpoint width cap')
        return (*ordinary,tuple(sorted((a,b) for a,b in mates.items() if a<b)))


def compile_compact(model,check=lambda:None,*,max_states=4096,max_steps=50000):
    if any(type(bound)is not int or bound<0 for bound in (max_states,max_steps)):raise ValueError('Invalid compact graph work bound.')
    stats=model.stats;zero=(0,)*model.n
    check()
    if model.missing:return {'complete':True,'supports':zero,'rows':0,'states':0,'transitions':0,'witness':None}
    layers=[{model.root_state:0}];nodes={};created=1;steps=0
    stats['compact_factor_states']+=1
    if created>max_states:raise CompactLimit('state cap')
    for level,codes in enumerate(model.choices):
        check();following={}
        for state,node in layers[-1].items():
            edges=[]
            for code in codes:
                check();steps+=1;stats['compact_factor_transitions']+=1
                if steps>max_steps:raise CompactLimit('transition cap')
                child=model.transition(level,state,code,check)
                if child is None:continue
                ident=following.get(child)
                if ident is None:
                    ident=created;created+=1;stats['compact_factor_states']+=1
                    if created>max_states:raise CompactLimit('state cap')
                    following[child]=ident
                else:stats['compact_factor_merges']+=1
                edges.append((code,ident))
            nodes[node]=(level,tuple(edges))
        layers.append(following)
        if not following:
            check();return {'complete':True,'supports':zero,'rows':0,'states':created,'transitions':steps,'witness':None}
    # Every node was reached by a legal prefix. A suffix path to a final node
    # therefore witnesses every labelled edge retained in this reverse pass.
    counts={node:1 for state,node in layers[-1].items()};supports=[0]*model.n
    for layer in reversed(layers[:-1]):
        check()
        for node in layer.values():
            level,edges=nodes[node];number=0
            for code,child in edges:
                check();stats['compact_factor_backward_edges']+=1
                if counts.get(child,0):supports[level]|=1<<code;number+=counts[child]
            counts[node]=number
    check();witness=[];node=0
    while len(witness)<model.n:
        check();level,edges=nodes[node]
        code,node=next((code,child) for code,child in edges if counts.get(child,0))
        witness.append(code)
    return {'complete':True,'supports':tuple(supports),'rows':counts[0],'states':created,'transitions':steps,'witness':tuple(witness)}


class CompactFactorPruner:
    def __init__(self,owner,*,limit=4,gold=False):
        if type(limit)is not int or not 1<=limit<=16:raise ValueError('Invalid compact factor cache bound.')
        if type(gold)is not bool or gold and owner.target!='gold':raise ValueError('Gold compilation requires the Gold objective.')
        self.gold=gold;self.owner=owner;self.problem=owner.problem;self.binding=source_binding(owner);self.limit=limit
        self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0);self.solution=None
    def prune(self,domains,check=lambda:None,*,max_placements=4096,max_frontier=96,max_resources=32,
              max_states=4096,max_steps=50000,max_ports=128):
        self.solution=None
        if source_binding(self.owner)!=self.binding:raise ValueError('Compact factor source binding changed.')
        limits=(max_placements,max_frontier,max_resources,max_states,max_steps,max_ports)
        if any(type(bound)is not int or bound<0 for bound in limits):raise ValueError('Invalid compact factor bound.')
        if (len(domains)!=self.problem.n or any(type(mask)is not int or mask<0 or
                mask&~sum(1<<code for code in set(self.problem.domains[cell])) for cell,mask in enumerate(domains))):
            raise ValueError('Invalid compact factor query domains.')
        if not all(domains):return False
        check();key=(tuple(domains),limits)
        if key in self.cache:
            self.stats['compact_factor_cache_hits']+=1;result=self.cache[key];self.cache.move_to_end(key)
        else:
            self.stats['compact_factor_queries']+=1
            self.stats['compact_gold_queries']+=int(self.gold)
            try:
                factory=GoldCompactFactors if self.gold else CompactFactors
                model=factory(self.problem,domains,check,stats=self.stats,max_placements=max_placements,
                              max_frontier=max_frontier,max_resources=max_resources,
                              **({'max_ports':max_ports} if self.gold else {}))
                result=compile_compact(model,check,max_states=max_states,max_steps=max_steps)
                if self.gold and result['witness'] is not None:
                    check();validation=self.problem.validate(result['witness']);self.stats['compact_gold_validations']+=1
                    if not validation.get('valid'):raise RuntimeError('Completed Gold factor witness failed independent validation.')
            except CompactLimit as error:
                self.stats['compact_factor_unknown']+=1;self.stats['compact_gold_unknown']+=int(self.gold)
                self.stats['compact_factor_admission_skips']+=int('admission' in str(error) or 'width' in str(error))
                result=None
            check();self.cache[key]=result
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
            if result is not None:
                self.stats['compact_factor_completed']+=1;self.stats['compact_gold_completed']+=int(self.gold)
        if result is None:return False
        current=[mask&support for mask,support in zip(domains,result['supports'])]
        check();self.stats['compact_factor_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,current))
        changed=current!=domains;domains[:]=current
        if self.gold:self.solution=result['witness']
        return changed
