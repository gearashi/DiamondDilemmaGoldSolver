"""Short GF(2) contradiction certificates checked from actual allowed patterns."""
from constraint_algebra import affine_equations
from constraint_relations import bits

STAT_KEYS=('affine_reason_equations','affine_reason_xors','affine_reason_pattern_checks','affine_reason_certificates','affine_reason_caps')


def _binding(problem):
    return (problem.fingerprint,tuple(tuple(row) for row in problem.domains),tuple(tuple(map(int,row)) for row in problem.masks),tuple(problem.edges))


def coordinates(problem):
    result=[[None]*33 for _ in range(problem.n)]
    for edge,(a,sa,b,sb) in enumerate(problem.edges):
        for port in range(11):result[a][11*sa+port]=11*edge+port;result[b][11*sb+10-port]=11*edge+port
    return result


def _domains(problem,domains):
    if not isinstance(domains,(tuple,list)) or len(domains)!=problem.n or any(type(mask)is not int or mask<=0 or mask&~sum(1<<v for v in row) for mask,row in zip(domains,problem.domains)):raise ValueError('Invalid affine proof domains.')


def verify_affine_conflict(problem,domains,certificate,check=lambda:None,stats=None):
    _domains(problem,domains);binding=_binding(problem);before=tuple(domains);stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    if type(certificate)is not dict or set(certificate)!={'equations'} or type(certificate['equations'])is not list or not 1<=len(certificate['equations'])<=8192:raise ValueError('Invalid affine certificate.')
    mapping=coordinates(problem)
    if any(any(coord is None for coord in row) for row in mapping):return False
    result=0;parity=0;seen=set()
    for item in certificate['equations']:
        check()
        if type(item)is not list or len(item)!=3:raise ValueError('Invalid local affine equation.')
        cell,raw,rhs=item
        if type(cell)is not int or not 0<=cell<problem.n or type(raw)is not str or not raw.startswith('0x') or len(raw)>11 or type(rhs)is not int or rhs not in (0,1):raise ValueError('Invalid affine equation fields.')
        try:mask=int(raw,16)
        except ValueError:raise ValueError('Invalid affine equation mask.') from None
        if mask<0 or mask>>33 or (cell,mask,rhs) in seen:raise ValueError('Invalid/repeated affine equation.')
        seen.add((cell,mask,rhs))
        for code in bits(domains[cell]):
            check();pattern=sum(int(value)<<(11*side) for side,value in enumerate(problem.masks[code]));stats['affine_reason_pattern_checks']+=1
            if (pattern&mask).bit_count()%2!=rhs:return False
        for local in bits(mask):result^=1<<mapping[cell][local]
        parity^=rhs;stats['affine_reason_equations']+=1
    check()
    if _binding(problem)!=binding or tuple(domains)!=before:raise ValueError('Affine proof source changed.')
    return result==0 and parity==1


def produce_affine_conflict(problem,domains,check=lambda:None,*,max_equations=8192,max_xors=200000,stats=None):
    _domains(problem,domains);binding=_binding(problem);before=tuple(domains);stats={} if stats is None else stats
    for key in STAT_KEYS:stats.setdefault(key,0)
    if type(max_equations)is not int or not 0<=max_equations<=8192 or type(max_xors)is not int or not 0<=max_xors<=2000000:raise ValueError('Invalid affine reason caps.')
    mapping=coordinates(problem)
    if any(any(coord is None for coord in row) for row in mapping):return None
    basis={};records=[];work=0
    for cell,domain in enumerate(domains):
        patterns=[sum(int(value)<<(11*side) for side,value in enumerate(problem.masks[code])) for code in bits(domain)]
        for mask,rhs in affine_equations(patterns,33):
            check()
            if len(records)>=max_equations:stats['affine_reason_caps']+=1;return None
            global_mask=0
            for local in bits(mask):global_mask^=1<<mapping[cell][local]
            lineage=1<<len(records);records.append([cell,hex(mask),rhs]);value=(global_mask<<1)|rhs
            while value>>1:
                check();work+=1;stats['affine_reason_xors']+=1
                if work>max_xors:stats['affine_reason_caps']+=1;return None
                pivot=value.bit_length()-1
                if pivot not in basis:basis[pivot]=(value,lineage);break
                prior,why=basis[pivot];value^=prior;lineage^=why
            if value==1:
                certificate={'equations':[records[i] for i in bits(lineage)]}
                if not verify_affine_conflict(problem,domains,certificate,check,stats):raise RuntimeError('Affine contradiction failed independent replay.')
                if _binding(problem)!=binding or tuple(domains)!=before:raise ValueError('Affine producer source changed.')
                stats['affine_reason_certificates']+=1;return certificate
    return None
