"""Exact, consistently oriented triangular mesh for the pentagonal bipyramid."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Cell:
    index: int
    face: int
    i: int
    j: int
    inverted: bool
    vertices: tuple[int, int, int]

@dataclass(frozen=True)
class Board:
    subdivision: int
    macro_faces: tuple
    vertices: tuple
    cells: tuple[Cell, ...]
    neighbors: tuple
    edges: tuple

    @property
    def neighbor_cells(self):
        return tuple(tuple(p[0] for p in row) for row in self.neighbors)

    @property
    def neighbor_sides(self):
        return tuple(tuple(p[1] for p in row) for row in self.neighbors)

    def as_dict(self):
        return {
            'subdivision': self.subdivision, 'macro_faces': self.macro_faces,
            'vertices': self.vertices,
            'cells': [vars(cell) for cell in self.cells],
            'neighbors': self.neighbors, 'edges': self.edges,
            'convention': 'side s runs vertex s to (s+1)%3; rotation adds to side',
        }

def build_board(n=4):
    """Faces 0..4 are top; 5..9 bottom. Each contains n*n cells.

    Macro vertex IDs: top=0, bottom=1, equator=2..6. Cell order increases
    face, i, j, upright then inverted. Local face vertices are chart points
    (0,0), (n,0), (0,n), clockwise in screen coordinates. Exact barycentric
    integer keys identify seam vertices without floating-point tolerances.
    """
    if type(n) is not int or n < 1:
        raise ValueError('subdivision must be a positive integer')
    faces = tuple((0, 2+k, 2+(k+1)%5) for k in range(5))
    faces += tuple((1, 2+(k+1)%5, 2+k) for k in range(5))
    vertices, vertex_ids, cells = [], {}, []
    def vertex(face, i, j):
        key = tuple(sorted((v,w) for v,w in zip(face, (n-i-j,i,j)) if w))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]
    for face_index, face in enumerate(faces):
        for i in range(n):
            for j in range(n-i):
                points = ((i,j), (i+1,j), (i,j+1))
                ids = tuple(vertex(face,*p) for p in points)
                cells.append(Cell(len(cells),face_index,i,j,False,ids))
                if i+j < n-1:
                    points = ((i+1,j), (i+1,j+1), (i,j+1))
                    ids = tuple(vertex(face,*p) for p in points)
                    cells.append(Cell(len(cells),face_index,i,j,True,ids))
    incidence = {}
    for cell in cells:
        for side in range(3):
            start,end = cell.vertices[side],cell.vertices[(side+1)%3]
            incidence.setdefault(tuple(sorted((start,end))), []).append((cell.index,side,start,end))
    neighbors = [[None]*3 for _ in cells]
    edges = []
    for occurrences in incidence.values():
        if len(occurrences) != 2:
            raise RuntimeError('boundary or nonmanifold edge')
        a,sa,va,wa = occurrences[0]
        b,sb,vb,wb = occurrences[1]
        if (va,wa) != (wb,vb):
            raise RuntimeError('inconsistent orientation')
        neighbors[a][sa] = (b,sb)
        neighbors[b][sb] = (a,sa)
        edges.append((a,sa,b,sb))
    if any(p is None for row in neighbors for p in row):
        raise RuntimeError('incomplete adjacency')
    return Board(n, faces, tuple(vertices), tuple(cells),
                 tuple(tuple(row) for row in neighbors), tuple(edges))

if __name__ == '__main__':
    import json
    print(json.dumps(build_board().as_dict(), indent=2))
