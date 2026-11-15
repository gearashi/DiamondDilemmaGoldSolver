"""Replay mixed-turn polygons against a fixed oriented triangular mesh."""
from collections import Counter


def canonical(cycle):
    """Return tile/incoming-side/outgoing-side triples at a canonical word origin."""
    back=tuple((tile,outside,inside) for tile,inside,outside in reversed(cycle))
    candidates=[]
    for direction in (tuple(cycle),back):
        turns=tuple((b-a)%3 for _,a,b in direction)
        for start in range(len(turns)):
            candidates.append((turns[start:]+turns[:start],direction[start:]+direction[:start]))
    return min(candidates)


class TurnWords:
    def __init__(self, problem, domains=None, lengths=10, max_steps=100000):
        self.problem=problem
        self.domains=domains
        self.lengths=lengths
        self.max_steps=max_steps
        self.table={}
        self.stats=Counter()

    def roots(self,word):
        if word in self.table:
            self.stats['word_reuses']+=1
            return self.table[word]
        self.stats['distinct_words']+=1
        roots=[]
        for start in range(self.problem.n):
            for heading in range(3):
                self.stats['root_walks']+=1
                row=[]
                seen=set()
                cell,arrival=start,heading
                for turn in word:
                    if cell<0 or cell in seen:
                        break
                    row.append((cell,arrival))
                    seen.add(cell)
                    cell,arrival=self.problem.board.neighbors[cell][(arrival+turn)%3]
                else:
                    if (cell,arrival)==(start,heading):
                        roots.append(tuple(row))
        self.table[word]=roots
        return roots

    def possible(self,cycle):
        word,cycle=canonical(cycle)
        roots=self.roots(word)
        if self.domains is None:
            return bool(roots)
        for row in roots:
            if all(self.domains[cell]>>(3*tile+(world-side)%3)&1
                   for (cell,world),(tile,side,_) in zip(row,cycle)):
                return True
        return False

    def conflicts(self,pairs,maximum=8):
        """Return only witnessed impossible polygons; incomplete walks cannot exclude."""
        neighbors=[[] for _ in range(self.problem.n)]
        for a,b in sorted(pairs):
            t,i=divmod(a,3);u,j=divmod(b,3)
            if t!=u:
                neighbors[t].append((u,i,j,(a,b)))
                neighbors[u].append((t,j,i,(a,b)))
        emitted=[]
        steps=0
        for start,adj in enumerate(neighbors):
            for second,first_side,in_side,first_pair in adj:
                if second<=start:
                    continue
                stack=[(second,in_side,(start,second),(),(first_pair,))]
                while stack:
                    steps+=1
                    if steps>self.max_steps:
                        self.stats['traversal_limit']+=1
                        return emitted
                    tile,arrival,visited,body,edges=stack.pop()
                    for next_tile,outgoing,incoming,pair in neighbors[tile]:
                        if outgoing==arrival:
                            continue
                        next_body=body+((tile,arrival,outgoing),)
                        if next_tile==start:
                            if len(visited)<3 or second>=tile or incoming==first_side:
                                continue
                            cycle=((start,incoming,first_side),)+next_body
                            if len({(b-a)%3 for _,a,b in cycle})==1:
                                continue
                            self.stats['mixed_polygons']+=1
                            if not self.possible(cycle):
                                self.stats['conflicts']+=1
                                emitted.append(edges+(pair,))
                                if len(emitted)>=maximum:
                                    return emitted
                        elif (next_tile>start and next_tile not in visited and len(visited)<self.lengths):
                            stack.append((next_tile,incoming,visited+(next_tile,),next_body,edges+(pair,)))
        return emitted

    def conditional(self,options,required):
        """Join both ends of a required path using one optional physical seam."""
        if not required:
            return set()
        forced=self.conflicts(required,maximum=1)
        if forced:
            self.stats['required_conflicts']+=1
            return set(options)
        adj=[[] for _ in range(self.problem.n)]
        for a,b in sorted(required):
            t,i=divmod(a,3);u,j=divmod(b,3)
            adj[t].append((u,i,j))
            adj[u].append((t,j,i))
        removed=set()
        steps=0
        for start,adjacent in enumerate(adj):
            for second,first_out,arrival in adjacent:
                stack=[(second,arrival,(start,second),())]
                while stack:
                    tile,incoming,visited,body=stack.pop()
                    steps+=1
                    if steps>self.max_steps:
                        self.stats['conditional_limit']+=1
                        return removed
                    if tile>start and len(visited)>=3:
                        for end_out in range(3):
                            if end_out==incoming:
                                continue
                            for start_in in range(3):
                                pair=3*start+start_in,3*tile+end_out
                                if start_in==first_out or pair not in options or pair in required or pair in removed:
                                    continue
                                polygon=((start,start_in,first_out),)+body+((tile,incoming,end_out),)
                                if len({(b-a)%3 for _,a,b in polygon})==1:
                                    continue
                                self.stats['conditional_polygons']+=1
                                if not self.possible(polygon):
                                    self.stats['conditional_exclusions']+=1
                                    removed.add(pair)
                    if len(visited)<self.lengths:
                        for u,outside,inside in adj[tile]:
                            if outside!=incoming and u not in visited:
                                stack.append((u,inside,visited+(u,),body+((tile,incoming,outside),)))
        return removed
