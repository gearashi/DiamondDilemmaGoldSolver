"""Rebuild sphere Euler weights using boundary-region and geographical-corner equivalence."""
from collections import Counter
from itertools import product


class Partition:
    def __init__(self,size):self.parent=list(range(size))
    def root(self,v):
        while self.parent[v]!=v:
            self.parent[v]=self.parent[self.parent[v]];v=self.parent[v]
        return v
    def join(self,x,y):self.parent[self.root(x)]=self.root(y)


def weights(tile):
    chord=[]
    for (s,u),(t,v) in tile['segments']:
        a,b=12*s+u,12*t+v
        if (a==b or not 0<a%12<12 or not 0<b%12<12):raise ValueError('Invalid chord endpoints.')
        chord.append(tuple(sorted((a,b))))
    ends=set(v for pair in chord for v in pair)
    if len(ends)!=2*len(chord) or any(a<c<b<d or c<a<d<b for a,b in chord for c,d in chord):
        raise ValueError('Expected embedded disjoint chords in a triangular disk.')
    # Glue the two incident boundary intervals on each side of every chord.
    regions=Partition(36)
    for v in range(36):
        if v not in ends:regions.join((v-1)%36,v)
    for a,b in chord:regions.join(a,(b-1)%36);regions.join((a-1)%36,b)
    faces={regions.root(v) for v in range(36)}
    if len(faces)!=len(chord)+1:raise ValueError('Boundary walks failed to reconstruct all face disks.')
    colors={};parity=0
    for v in range(36):
        if v in ends:parity^=1
        face=regions.root(v)
        if colors.setdefault(face,parity)!=parity:raise ValueError('Inconsistent face colors.')
    result=[]
    for phase in (0,1):
        dark=[colors[regions.root(v)]^phase for v in range(36)]
        corners=sum(dark[i] for i in (0,12,24))
        runs=sum(dark[i] for i in range(36) if i%12==0 or i in ends)
        faces_dark=sum(c^phase for c in colors.values())
        result.append(corners-3*runs+6*faces_dark)
    return tuple(result)


def vertex_corrections(n,edges,strands):
    mesh=Partition(3*n)
    for a,s,b,t in edges:
        mesh.join(3*a+s,3*b+(t+1)%3)
        mesh.join(3*a+(s+1)%3,3*b+t)
    degree=Counter(mesh.root(i) for i in range(3*n))
    if len(degree)-len(edges)+n!=2:raise ValueError('Expected a closed triangulated sphere.')
    exceptions=[v for v,d in degree.items() if d!=6]
    values=set()
    for coloring in product((0,1),repeat=len(exceptions)):
        # Summing edge endpoint-color differences over the sphere leaves
        # exactly the colors of odd-degree vertices and the number of strands.
        if sum(degree[v]*c for v,c in zip(exceptions,coloring))%2!=strands%2:continue
        values.add(6-sum((6-degree[v])*c for v,c in zip(exceptions,coloring)))
    return sorted(values),dict(sorted(Counter(degree.values()).items()))


def reconstruct(problem):
    rows=[weights(tile) for tile in problem.tiles]
    allowed,degree=vertex_corrections(problem.n,problem.edges,problem.total_segments)
    return rows,allowed,degree
