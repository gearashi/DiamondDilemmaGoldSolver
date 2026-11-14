"""Immutable assignment and strand facts for native CUDA lane donation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PropagatedPrefix:
    source: object
    assignments: tuple
    unions: tuple
    supports: tuple

    @classmethod
    def capture(cls, engine, column):
        o = engine.offsets
        n = int(column[9])
        cells = column[o["assign_log"]:o["assign_log"]+n]
        assignments = tuple((int(c),int(column[o["board"]+c])) for c in cells)
        unions = tuple((int(column[o["union_child"]+i]),int(column[o["union_parent"]+i]))
                       for i in range(int(column[10]))) if engine.incremental_strands else ()
        supports = ()
        if engine.support_reuse:
            entries = []
            for field, size in (("support_keys",engine.support_slots),("support_values",3*engine.support_slots)):
                entries.append((o[field],column[o[field]:o[field]+size].tobytes()))
            supports = tuple(entries)
        return cls(engine,assignments,unions,supports)

    def install(self, engine, target, cube, assigned, joined):
        """Rebase only the facts true *before* selecting this frame's child."""
        if engine is not self.source:
            raise ValueError("A donated prefix belongs to its source CUDA engine")
        if (type(assigned) is not int or not 0 <= assigned <= len(self.assignments)
                or type(joined) is not int or not 0 <= joined <= len(self.unions)):
            raise ValueError("Donation rollback mark exceeds its captured log")
        o = engine.offsets
        used_cells = set()
        used_tiles = set()
        for i,(cell,code) in enumerate(self.assignments[:assigned]):
            if (not 0 <= cell < engine.n or not 0 <= code < engine.nc
                    or cell in used_cells or code//3 in used_tiles):
                raise ValueError("Invalid donated placement or duplicate owner")
            used_cells.add(cell)
            used_tiles.add(code//3)
            if cube[cell] != 1<<code:
                raise ValueError("Donated prefix assignment is outside its child domain")
            target[o["board"]+cell] = code
            target[o["owner"]+code//3] = cell
            target[o["assign_log"]+i] = cell
        target[9] = assigned
        for i,(child,parent) in enumerate(self.unions[:joined]):
            if (not 0 <= child < 16*engine.n or not 0 <= parent < 16*engine.n or child == parent
                    or child//16 not in used_cells or parent//16 not in used_cells
                    or target[o["parent"]+child] != child or target[o["parent"]+parent] != parent):
                raise ValueError("Donation union escapes its assigned rollback prefix")
            target[o["union_child"]+i] = child
            target[o["union_parent"]+i] = parent
            target[o["union_size"]+i] = target[o["size"]+parent]
            target[o["parent"]+child] = parent
            target[o["size"]+parent] += target[o["size"]+child]
        target[10] = joined
        for offset,payload in self.supports:
            words=np.frombuffer(payload,dtype=np.uint32)
            target[offset:offset+len(words)] = words


def donation_prefixes(engine, column, parts):
    """Parallel to remaining_domain_cubes, with a rollback mark for each box."""
    if int(column[0]) != 0:
        raise ValueError('Only an active donor can transfer unfinished descendants')
    depth=int(column[1])
    o=engine.offsets
    marks=[]
    if int(column[6]) not in (3,4):
        if all(engine._mask(column[:,None],0,cell) for cell in range(engine.n)):
            marks.append((int(column[9]),int(column[10])))
    for level in range(depth-1,-1,-1):
        start=o["frame_remaining"]+level*engine.words
        if np.any(column[start:start+engine.words]):
            marks.append((int(column[o["frame_assign"]+level]),int(column[o["frame_union"]+level])))
    if len(marks)!=len(parts):
        raise ValueError("Donation ownership and rollback marks disagree")
    return marks

