"""Bounds and changed-edge prices for the signed physical repair trajectory."""
import hashlib
import json
import numpy as np


class RefitBounds:
    def __init__(self):
        self.history = {}

    def upper(self, bonds):
        return min([len(bonds), *(cap+len(bonds-old) for old,cap in self.history.items())])

    def record(self, bonds, ceiling):
        if 0 <= ceiling < len(bonds):
            key=frozenset(bonds)
            if len(self.history)>=32 and key not in self.history:
                del self.history[next(iter(self.history))]
            self.history[key]=min(ceiling,self.history.get(key,len(key)))


    def snapshot(self, fingerprint, mandatory):
        rows=[[sorted(bonds),cap] for bonds,cap in self.history.items()]
        raw=json.dumps([fingerprint,sorted(mandatory),rows],separators=(',',':')).encode()
        return dict(rows=rows,checksum=hashlib.sha256(raw).hexdigest())

    def restore(self, saved, fingerprint, mandatory):
        if not isinstance(saved,dict):
            return
        try:
            rows=saved['rows']
            if not isinstance(rows,list) or len(rows)>32:
                return
            raw=json.dumps([fingerprint,sorted(mandatory),rows],separators=(',',':')).encode()
            if hashlib.sha256(raw).hexdigest()!=saved['checksum']:
                return
            for bonds,cap in rows:
                if type(cap) is not int or not isinstance(bonds,list) or not 0<=cap<len(bonds):
                    return
                if any(len(b)!=2 or any(type(x) is not int or x<0 for x in b) or b[0]>=b[1] for b in bonds):
                    return
            for bonds,cap in rows:
                self.record(frozenset(map(tuple,bonds)),cap)
        except (KeyError,TypeError,ValueError):
            return


def changed_dart_prices(prior, current, dart):
    template = current.geometry.templates.get(dart)
    row = prior.prices.get(dart)
    if row is None or template is None or prior.geometry is not current.geometry:
        return None
    if not hasattr(current,'_refit_changes'):
        current._refit_changes = prior.dart_phases != current.dart_phases
        current._reversed_or_identical = not current._refit_changes.any() or current._refit_changes.all()
    if current._reversed_or_identical:
        current.stats['whole_dart_rows_shared']+=1
        return row
    changed = current._refit_changes
    geometry=current.geometry
    if not hasattr(geometry,'refit_candidates'):
        by_mask={}
        for side, mask in enumerate(geometry.masks):
            by_mask.setdefault(int(mask),[]).append(side)
        geometry.refit_by_mask={m:np.asarray(values,dtype=int) for m,values in by_mask.items()}
        geometry.refit_candidates={}
        current.stats['delta_mask_index_rows']+=len(geometry.masks)
    eligible=geometry.refit_candidates.get(dart)
    if eligible is None:
        mask=int(current.model.reverse[geometry.masks[dart]])
        candidates=geometry.refit_by_mask.get(mask,np.empty(0,dtype=int))
        eligible=candidates[template[candidates]>=0]
        geometry.refit_candidates[dart]=eligible
    flips = eligible[changed[eligible] != changed[dart]]
    current.stats['delta_dart_rows'] += 1
    current.stats['delta_price_bits_examined'] += len(eligible)
    current.stats['delta_price_bits_changed'] += len(flips)
    if not len(flips):
        return row
    result = row.copy()
    result[flips] ^= True
    return result


def changed_score(signed, board, last_board, old_score, previous=None):
    """Reprice only changed placements and endpoints on opposite sides of a phase flip."""
    model=signed.model
    changed=np.flatnonzero(board!=last_board)
    if not len(changed) and previous is None:
        signed.stats['score_replays_avoided']+=1
        return old_score
    if previous is None and len(changed)*6<=model.n:
        incident=getattr(model,'_signed_incident_edges',None)
        if incident is None:
            incident=[[] for _ in range(model.n)]
            for edge,(a,b) in enumerate(zip(model.a,model.b)):
                incident[a].append(edge)
                incident[b].append(edge)
            model._signed_incident_edges=incident
        edges=np.asarray(sorted({edge for cell in changed for edge in incident[cell]}),dtype=int)
    else:
        moved=board!=last_board
        affected=moved[model.a] | moved[model.b]
        if previous is not None:
            if previous.geometry is not signed.geometry:
                return signed.score(board)
            flips=np.asarray(previous.phases) != np.asarray(signed.phases)
            cells=flips[board//3]
            affected |= cells[model.a] != cells[model.b]
        edges=np.flatnonzero(affected)
        if not len(edges):
            signed.stats['score_replays_avoided']+=1
            return old_score
    if len(edges)*2 > len(model.a):
        signed.stats['full_score_edges']+=len(model.a)
        return signed.score(board)
    signed.stats['delta_score_edges']+=len(edges)
    def contributions(engine,values):
        l=values[model.a[edges]]
        r=values[model.b[edges]]
        ls=model.sa[edges]
        rs=model.sb[edges]
        ok=engine.labels[l,ls] == engine.reverse[engine.labels[r,rs]]
        raw=model.masks[l,ls] == model.reverse[model.masks[r,rs]]
        a=signed.sides[l,ls]
        b=signed.sides[r,rs]
        banned=np.fromiter(((min(x,y),max(x,y)) in signed.forbid for x,y in zip(a,b)),bool,len(a))
        return int(np.count_nonzero(ok & ~banned)),int(np.count_nonzero(raw & banned))
    before=contributions(previous if previous is not None else signed,last_board)
    after=contributions(signed,board)
    return tuple(x+b-a for x,a,b in zip(old_score,before,after))
