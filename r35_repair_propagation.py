"""Project geographic pose-support propagation into necessary changes of a named board."""
from collections import deque
from functools import lru_cache
from r27_structure_filters import domain_bits


class RepairPropagation:
    def __init__(self,problem,mates):
        self.p=problem
        self.root=domain_bits(problem)
        self.neighbors=[tuple((int(b),int(j)) for b,j in zip(row,sides))
                        for row,sides in zip(problem.neighbors,problem.sides)]
        self.support={}
        for s in range(3):
            for j in range(3):
                rows=[]
                for code in range(3*problem.n):
                    tile,turn=divmod(code,3)
                    dart=3*tile+(s-turn)%3
                    rows.append(sum(1<<(3*(b//3)+(j-b%3)%3) for b in mates[dart]))
                self.support[s,j]=rows
        @lru_cache(maxsize=16384)
        def supported(side,facing,domain):
            masks=self.support[side,facing]
            found=0
            while domain:
                bit=domain&-domain
                domain-=bit
                found|=masks[bit.bit_length()-1]
            return found
        self.supported=supported

    def narrow(self,seed,anchors,max_domain=40,unique=True):
        current=list(self.root)
        queue=deque()
        owned={}
        for c in anchors:
            pose=1<<seed[c]
            if not pose&current[c]:return None
            current[c]=pose
        for c,d in enumerate(current):
            if not d:return None
            if d.bit_count()<=max_domain:queue.append(c)
        while queue:
            c=queue.popleft()
            d=current[c]
            if unique:
                tile=((d&-d).bit_length()-1)//3
                tile_bits=7<<(3*tile)
                if d&~tile_bits==0 and tile not in owned:
                    owned[tile]=c
                    for b,other in enumerate(current):
                        if b!=c and other&tile_bits:
                            revised=other&~tile_bits
                            if not revised:return None
                            current[b]=revised
                            if revised.bit_count()<=max_domain:queue.append(b)
            for s,(b,j) in enumerate(self.neighbors[c]):
                if b<0:continue
                revised=current[b]&self.supported(s,j,d)
                if not revised:return None
                if revised!=current[b]:
                    current[b]=revised
                    if revised.bit_count()<=max_domain:queue.append(b)
        return current

    def project(self,state,anchors,max_domain=40,unique=True):
        fixed=frozenset(anchors)
        if state.covered(fixed):return
        state.work['propagation_anchors']+=1
        ds=self.narrow(state.codes,fixed,max_domain,unique)
        if ds is None:
            state.add(fixed,'propagated_anchor_conflict')
            return
        for cell,d in enumerate(ds):
            if not d&(1<<state.codes[cell]):
                state.add(fixed|{cell},'propagated_seed_conflict')

    def anchor_sets(self,radius):
        for c in range(self.p.n):yield (c,)
        if radius<=0:return
        for c in range(self.p.n):
            seen={c}
            ring={c}
            for _ in range(radius):
                ring={b for a in ring for b,_ in self.neighbors[a] if b>=0}-seen
                seen.update(ring)
            for b in sorted(seen):
                if b>c:yield (c,b)
