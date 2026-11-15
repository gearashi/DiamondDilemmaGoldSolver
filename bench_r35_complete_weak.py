"""Complete SCIP's exact weak linear rows using globally valid one-variable input bounds.

The resulting version-1.0 file is checked independently by the upstream VIPR checker.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def number(value):
    return Fraction(value) if '/' in value else int(value)


def expression(line):
    tokens = line.split(' { ', 1)[0].split()
    sense, rhs, n = tokens[1], number(tokens[2]), int(tokens[3])
    if len(tokens) != 4+2*n:
        raise ValueError('Invalid linear axiom or derivation.')
    return sense, rhs, {int(tokens[4+2*i]):number(tokens[5+2*i]) for i in range(n)}


def derivations(source):
    line = source.readline()
    while line and not line.startswith('CON '):
        yield None,line
        line=source.readline()
    yield None,line
    count=int(line.split()[1])
    for i in range(count):
        yield i,source.readline()
    while True:
        line=source.readline()
        if not line:
            raise ValueError('Missing derivations.')
        yield None,line
        if line.startswith('DER '):
            break
    for i,line in enumerate(source,count):
        yield i,line


WEAK = ' { lin weak { 0 } '


def weak_terms(line):
    body = line.split(WEAK,1)[1].split('}',1)[0].split()
    terms = [(int(i),number(w)) for i,w in zip(body[1::2],body[2::2])]
    if int(body[0])!=len(terms):
        raise ValueError('Malformed weak linear combination.')
    return terms


def complete(source, output):
    dependencies=set()
    with source.open(encoding='ascii') as file:
        for line in file:
            if WEAK in line:
                dependencies.update(i for i,_ in weak_terms(line))
            elif ' { lin weak ' in line:
                raise ValueError('Local weak derivations need the full upstream VIPR completion program.')

    expressions={}
    bounds=defaultdict(list)
    with source.open(encoding='ascii') as file:
        in_header=True
        for i,line in derivations(file):
            if line.startswith('RTP '):
                in_header=False
            published_bound = line.rstrip().endswith(' global')
            if i is None or (not in_header and i not in dependencies and not published_bound):
                continue
            sense,rhs,row=expression(line)
            if i in dependencies:
                expressions[i]=(rhs,row)
            if (in_header or published_bound) and len(row)==1:
                (variable,coef),=row.items()
                bounds[variable].append((i,sense,rhs,coef))

    replaced=0
    with source.open(encoding='ascii') as file, output.open('w',encoding='ascii',newline='\n') as out:
        for current,line in derivations(file):
            if WEAK not in line:
                out.write(line)
                continue
            sense,rhs,row=expression(line)
            terms=weak_terms(line)
            for i,weight in terms:
                other_rhs,other=expressions[i]
                rhs-=weight*other_rhs
                for v,c in other.items():
                    row[v]=row.get(v,0)-weight*c
            for v,c in row.items():
                if not c:
                    continue
                choices=[]
                for i,direction,side,coefficient in bounds[v]:
                    if i >= current:
                        continue
                    multiplier=Fraction(c,coefficient)
                    sign=direction if multiplier>0 else {'L':'G','G':'L','E':'E'}[direction]
                    if sign in (sense,'E'):
                        choices.append((multiplier*side,i,multiplier))
                if not choices:
                    raise ValueError('No valid input bound for a weak derivation.')
                bound,i,multiplier=(min(choices) if sense=='L' else max(choices))
                rhs-=bound
                terms.append((i,multiplier))
            if (sense=='L' and rhs<0) or (sense=='G' and rhs>0) or (sense=='E' and rhs):
                raise ValueError('Input bounds do not prove this weakened inequality.')
            prefix=line.split(WEAK,1)[0]
            suffix=line.rsplit('}',1)[1]
            out.write(prefix+' { lin '+str(len(terms))+' '+' '.join(f'{i} {w}' for i,w in terms)+' }'+suffix)
            replaced+=1
    return replaced


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    print(complete(**vars(parser.parse_args())))
