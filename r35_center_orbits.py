"""Proper, domain-preserving board rotations for physical cache identities and witness transport."""
import numpy as np


class CenterOrbits:
    def __init__(self,model):
        from r29_exchange_route import legal_cell_rotations
        from constraint_symmetry import proper_rotations
        self.model=model
        identity=tuple(range(model.n))
        from geometry import Board
        cells=({tuple(r) for r in legal_cell_rotations(model)} if isinstance(model.problem.board,Board) else {identity})
        self.transforms=[]
        if cells!={identity}:
            self.transforms=[t for t in proper_rotations(model.problem.board) if t.cells in cells]
        self.inverse=np.array([np.argsort(t.cells) for t in self.transforms],dtype=int)
        if self.transforms and len(set(self.inverse[:,0]))!=len(self.transforms):
            raise ValueError('Proper mesh rotations unexpectedly stabilize cell zero.')

    def canonical(self,tiles):
        values=np.asarray(tiles,dtype=int)
        if not self.transforms:return tuple(map(int,values)),None
        # Every position feeding canonical cell zero is distinct, and every
        # physical tile appears once. The first label determines the least image.
        transform=int(np.argmin(values[self.inverse[:,0]]))
        return tuple(map(int,values[self.inverse[transform]])),transform

    def restore(self,codes,transform):
        if transform is None:return list(map(int,codes))
        row=self.transforms[transform]
        return [3*(int(codes[d])//3)+(int(codes[d])%3-off)%3 for d,off in zip(row.cells,row.side_offsets)]

    def forward(self,codes,transform):
        if transform is None:return list(map(int,codes))
        return self.transforms[transform].apply(list(map(int,codes)))

    def key(self,tiles):
        from r29_exchange_route import physical_hash
        representative,transform=self.canonical(tiles)
        return physical_hash(representative),transform

    def cap(self,db,bound,domain,swaps,compute):
        """Existing unrotated rows stay readable; a verified rotation preserves an upper bound."""
        from r29_exchange_route import physical_hash
        from r30_bounds import defect_packings
        tiles=tuple(map(int,bound.tiles))
        representative,_=self.canonical(tiles)
        prefix=f'{domain}:{swaps}:'
        base=prefix+physical_hash(tiles);canonical=prefix+physical_hash(representative)
        saved=db.execute('SELECT min(cap) FROM r30_budget WHERE cap_key IN (?,?)',(base,canonical)).fetchone()[0]
        if saved is None and self.transforms:
            keys=[prefix+physical_hash(np.asarray(tiles)[inverse]) for inverse in self.inverse]
            marks=','.join('?' for _ in keys)
            saved=db.execute('SELECT min(cap) FROM r30_budget WHERE cap_key IN ('+marks+')',keys).fetchone()[0]
        if saved is None:saved=compute(bound,defect_packings(bound),swaps)
        db.execute('INSERT OR IGNORE INTO r30_budget VALUES(?,?)',(canonical,saved))
        return saved

    def lookup(self,db,binding,seed):
        """Recover an ordinary exact matched-seam witness in the requested coordinate frame."""
        import json
        if not self.transforms:return None
        values=np.asarray(seed,dtype=int)//3
        # Old optima are keyed in the coordinate system where the endpoint was solved.
        keys={binding+':'+','.join(map(str,values[inverse])):i for i,inverse in enumerate(self.inverse)}
        marks=','.join('?' for _ in keys)
        for key,raw in db.execute('SELECT key,result FROM optima WHERE key IN ('+marks+')',list(keys)):
            row=json.loads(raw)
            if not row.get('codes'):continue
            try:
                result=self.restore(row['codes'],keys[key])
                if tuple(c//3 for c in result)!=tuple(values):continue
                self.model.legal_board(result)
            except (IndexError,TypeError,ValueError):continue
            independent=self.model.problem.validate(result)
            if independent['matched_edges']!=row.get('score'):continue
            return dict(score=independent['matched_edges'],codes=result,components=independent['line_components'],
                        gold=independent['valid'])
        return None
