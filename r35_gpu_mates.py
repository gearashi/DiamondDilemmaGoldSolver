"""Stateless lookup of compulsory physical-component poses on the supplied classic DFS mesh."""
import numpy as np


INVALID=0xffffffff


def pose_table(neighbors,opposite,required,max_words=2000000):
    n=len(neighbors)
    if any(int(v)<0 for row in neighbors for v in row):return np.empty(0,np.uint32)
    adjacency=[[] for _ in range(n)]
    for a,b in required:
        t,s=divmod(a,3);u,j=divmod(b,3)
        adjacency[t].append((s,u,j));adjacency[u].append((j,t,s))
    if not any(adjacency):return np.empty(0,np.uint32)
    data=[0]*(n+(n+31)//32)
    for tile,row in enumerate(adjacency):
        if row:data[n+tile//32]|=1<<(tile%32)
    unseen={v for v,row in enumerate(adjacency) if row}
    while unseen:
        root=min(unseen);group={root};pending=[root]
        while pending:
            for _,v,_ in adjacency[pending.pop()]:
                if v not in group:group.add(v);pending.append(v)
        unseen-=group
        direct_only=len(data)+3*n*len(group)*(len(group)-1)>max_words
        for tile in sorted(group):
            relatives=sorted({v for _,v,_ in adjacency[tile]} if direct_only else group-{tile})
            if not relatives:continue
            data[tile]=len(data)
            data.append(len(relatives))
            for c in range(n):
                for rotation in range(3):
                    poses={tile:(c,rotation)};occupancy={c:tile}
                    stack=[tile];valid=True
                    while stack and valid:
                        t=stack.pop();place,rot=poses[t]
                        for side,u,j in adjacency[t]:
                            edge=(side+rot)%3
                            nc=int(neighbors[place][edge]);nr=(int(opposite[place][edge])-j)%3
                            if (u in poses and poses[u]!=(nc,nr)) or (nc in occupancy and occupancy[nc]!=u):
                                valid=False;break
                            if u not in poses:
                                poses[u]=(nc,nr);occupancy[nc]=u
                                if not direct_only:stack.append(u)
                    data.extend(((poses[u][0]<<9)|(3*u+poses[u][1]) for u in relatives) if valid
                                else (INVALID for _ in relatives))
    return np.array(data,np.uint32)


def host_conflict(table,board,cell,code,neighbors,opposite,faces,reverse):
    if not table.size:return -1
    def demands(target):
        for anchor in neighbors[target]:
            anchor=int(anchor);owner=int(board[anchor])
            if anchor==cell or owner<0:continue
            h=int(table[owner//3])
            if not h:continue
            count=int(table[h]);s=h+1+(3*anchor+owner%3)*count
            for packed in table[s:s+count]:
                val=int(packed)
                if val!=INVALID and val>>9==target:yield anchor,val&511
    header=int(table[code//3])
    if not header:return -1
    size=int(table[header]);start=header+1+(3*cell+code%3)*size
    for packed in table[start:start+size]:
        val=int(packed)
        if val==INVALID:return cell
        actual_cell,expected=divmod(val,512)
        occupant=board[actual_cell]
        if occupant>=0:
            if occupant!=expected:return actual_cell
            continue
        for anchor,other in demands(actual_cell):
            if other!=expected:return anchor
        for where,other in enumerate(board):
            if other>=0 and other//3==expected//3:return where
        for side in range(3):
            outside=int(neighbors[actual_cell][side]);other=board[outside]
            if outside!=cell and other>=0 and faces[expected,side]!=reverse[other,int(opposite[actual_cell][side])]:
                return outside
    return -1
