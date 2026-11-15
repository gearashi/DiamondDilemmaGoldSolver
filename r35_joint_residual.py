"""Reuse independently checked optima of finite physical donor/rotation repair models."""
from collections import Counter,OrderedDict
import base64
import hashlib
import json
import struct
import zlib

import numpy as np


class JointResidualCache:
    def __init__(self,capacity=256,max_bytes=8<<20):
        self.capacity=capacity
        self.max_bytes=max_bytes
        self.rows=OrderedDict()
        self.work=Counter()
        self.occupied=0

    def snapshot(self,scope):
        image=bytearray()
        for key,(labels,gain,branches) in self.rows.items():
            image.extend(struct.pack('<IIiq',len(key),len(labels),gain,branches))
            image.extend(key)
            image.extend(labels)
        work=dict(self.work)
        header=json.dumps([1,scope,work],sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(header+image).hexdigest()
        return dict(format=1,scope=scope,sha256=digest,work=work,
                    payload=base64.b64encode(zlib.compress(image,1)).decode('ascii'))

    def restore(self,scope,receipt):
        if receipt.get('format')!=1 or receipt.get('scope')!=scope:
            raise ValueError('Foreign joint-repair cache receipt.')
        limit=self.max_bytes+20*self.capacity
        if len(receipt['payload'])>4*((limit+max(1024,limit//100)+2)//3):
            raise ValueError('Oversized compressed joint-repair receipt.')
        counters=receipt['work']
        if not isinstance(counters,dict) or any(type(v) is not int or v<0 for v in counters.values()):
            raise ValueError('Invalid joint-repair work receipt.')
        header=json.dumps([1,scope,counters],sort_keys=True,separators=(',',':')).encode()
        try:
            blob=base64.b64decode(receipt['payload'],validate=True)
            decoder=zlib.decompressobj()
            image=decoder.decompress(blob,limit+1)
        except (ValueError,zlib.error) as exc:
            raise ValueError('Invalid compressed joint-repair cache receipt.') from exc
        if len(image)>limit or not decoder.eof or decoder.unused_data or hashlib.sha256(header+image).hexdigest()!=receipt['sha256']:
            raise ValueError('Invalid joint-repair cache receipt.')
        rows=OrderedDict()
        index=0
        size=0
        while index<len(image):
            if index+20>len(image):raise ValueError('Incomplete joint-repair cache record.')
            nk,nl,gain,branches=struct.unpack_from('<IIiq',image,index)
            index+=20
            if nk==0 or nl==0 or nl%2 or index+nk+nl>len(image) or branches<0 or gain<0:raise ValueError('Invalid cached joint-repair dimensions.')
            key=image[index:index+nk]
            labels=image[index+nk:index+nk+nl]
            choices=np.frombuffer(labels,dtype='<u2')
            if len(key)<8 or struct.unpack_from('<H',key,6)[0]!=len(choices):
                raise ValueError('Cached residual and witness dimensions differ.')
            if key in rows or sorted((choices//3).tolist())!=list(range(len(choices))):
                raise ValueError('Invalid cached joint-repair donors.')
            rows[key]=(labels,gain,branches)
            index+=nk+nl
            size+=nk+nl
        if size>self.max_bytes or len(rows)>self.capacity:
            raise ValueError('Oversized joint-repair receipt.')
        self.rows,self.occupied,self.work=rows,size,Counter(receipt['work'])

    def describe(self,model,board,cells,limit,disjoint,forbidden):
        cells=np.array(sorted(set(map(int,cells))),dtype=int)
        if any(c<0 or c>=model.n or c in model.problem.fixed for c in cells):
            raise ValueError('Expected a pool of movable cells.')
        n=len(cells)
        local={int(c):i for i,c in enumerate(cells)}
        tiles=board[cells]//3
        offsets=board[cells]%3
        # The labels are relative to the old rotation at the destination.
        codes=3*tiles[None,:,None]+(offsets[:,None,None]+np.arange(3))%3
        legal=model.legal[cells[:,None,None],codes].copy()
        if forbidden is not None:
            for c,v,neighbor in forbidden.blocked_codes(board):
                if c in local and neighbor not in local:
                    i=local[c]
                    legal[i] &= codes[i]!=v
        gains=np.zeros((n,n,3),dtype=np.int8)
        degrees=[]
        internal=[]
        for i,c in enumerate(cells):
            degrees.append(sum(int(nb) in local for nb in model.problem.neighbors[c]))
            for s,nb in enumerate(model.problem.neighbors[c]):
                if nb<0:continue
                t=model.problem.sides[c,s]
                if nb in local:
                    if i<local[nb]:internal.append((i,s,local[nb],t))
                    if limit!=0:continue
                gains[i]+=(model.masks[codes[i],s]==model.reverse[model.masks[board[nb],t]])
            gains[i]-=gains[i,i,0]
        # Assign comparable slot/donor labels using local numeric structure.
        # Deterministic ties can miss an isomorphic query; the complete finite
        # tables checked below protect every hit regardless of the label order.
        colors=[]
        values=(2*gains+legal).astype(np.int8)
        for i in range(n):
            colors.append((degrees[i],tuple(values[i,i]),tuple(sorted(map(tuple,values[i]))),
                           tuple(sorted(map(tuple,values[:,i])))))
        order=sorted(range(n),key=lambda i:(colors[i],int(tiles[i])))
        inv={old:new for new,old in enumerate(order)}
        pieces=[struct.pack('<i??H',limit,disjoint,forbidden is not None,n),
                legal[order][:,order].tobytes(),gains[order][:,order].tobytes()]
        baseline_legal=bool(legal[np.arange(n),np.arange(n),0].all())
        shared=[]
        for i,s,j,t in internal:
            if inv[i]>inv[j]:
                i,j,s,t=j,i,t,s
            a=codes[i,order].reshape(-1,1)
            b=codes[j,order].reshape(1,-1)
            pair=[struct.pack('<HH',inv[i],inv[j])]
            if limit!=0:
                match=model.masks[a,s]==model.reverse[model.masks[b,t]]
                pair.append(np.packbits(match).tobytes())
            if forbidden is not None:
                barred=forbidden.edge_bans(a,s,b,t)
                baseline_legal &= not barred[3*inv[i],3*inv[j]]
                pair.append(np.packbits(barred).tobytes())
            shared.append((inv[i],inv[j],b''.join(pair)))
        pieces.extend(part for i,j,part in sorted(shared))
        return b''.join(pieces),cells[order],tiles[order],offsets[order],baseline_legal

    def solve(self,solver,model,values,cells,**options):
        board=model.legal_board(values)
        limit=options.get('contacts')
        disjoint=bool(options.get('disjoint',False))
        if limit is None:limit=-1 if disjoint else int(options.get('adjacent',False))
        if type(limit) is not int or limit< -1:raise ValueError('Contact limit must be nonnegative or -1 for unrestricted.')
        if not len(cells):return solver(model,board,cells,**options)
        key,slots,tiles,offsets,baseline_legal=self.describe(model,board,cells,limit,disjoint,options.get('forbidden'))
        cached=self.rows.get(key)
        if cached is not None:
            labels,gain,branches=cached
            out=board.copy()
            choices=np.frombuffer(labels,dtype='<u2')
            out[slots]=3*tiles[choices//3]+(offsets+choices%3)%3
            model.legal_board(out)
            changed=set(int(c) for c in slots if out[c]!=board[c])
            degrees=Counter()
            for c in changed:
                degrees[c]=sum(int(nb) in changed for nb in model.problem.neighbors[c])
            if (limit>=0 and sum(degrees.values())//2>limit) or (disjoint and any(n>1 for n in degrees.values())):
                raise ArithmeticError('Cached joint witness violates the requested contact policy.')
            forbidden=options.get('forbidden')
            if forbidden is not None:
                incident=np.isin(model.a,slots)|np.isin(model.b,slots)
                if forbidden.violations(out)[incident].any():raise ArithmeticError('Cached joint witness violates physical mate rules.')
            if int(model.matches(out).sum()-model.matches(board).sum())!=gain:
                raise ArithmeticError('Cached joint witness failed its physical gain check.')
            self.rows.move_to_end(key)
            self.work['proved_hits']+=1
            self.work['original_branches_bypassed']+=branches
            return out,dict(status='OPTIMAL',branches=0)
        self.work['models_built']+=1
        out,info=solver(model,board,cells,**options)
        if info['status']=='OPTIMAL' and baseline_legal:
            index={int(t):i for i,t in enumerate(tiles)}
            labels=np.array([3*index[int(out[c]//3)]+(out[c]%3-offset)%3 for c,offset in zip(slots,offsets)],dtype='<u2').tobytes()
            amount=len(key)+len(labels)
            if self.capacity>0 and amount<=self.max_bytes:
                while self.rows and (len(self.rows)>=self.capacity or self.occupied+amount>self.max_bytes):
                    previous,(witness,_,_)=self.rows.popitem(last=False)
                    self.occupied-=len(previous)+len(witness)
                self.rows[key]=(labels,int(model.matches(out).sum()-model.matches(board).sum()),info['branches'])
                self.occupied+=amount
                self.work['optima_admitted']+=1
                self.work['peak_payload_bytes']=max(self.work['peak_payload_bytes'],self.occupied)
        return out,info
