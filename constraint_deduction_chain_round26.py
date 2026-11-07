"""Independent finite-domain deduction replay for CPU/GPU proof transport.

Every removal is re-established from the original puzzle and already checked
steps. Native reasons, producer hashes, provenance tags and unfinished traces
are never proof authority. Root restrictions remain explicit conclusion guards.
"""
from __future__ import annotations
import hashlib,itertools,json
from validator import reverse_mask
from constraint_knowledge import revalidate_witness

FORMAT='diamond-deduction-chain-v1'
MAX_EVENTS=2048
MAX_BYTES=1024*1024
CHAIN_STAT_KEYS=('chain_events','chain_removed_values','chain_seam_placements','chain_hall_placements','chain_loop_checks',
 'chain_affine_reason_equations','chain_affine_reason_xors','chain_affine_reason_pattern_checks','chain_affine_reason_certificates','chain_affine_reason_caps',
 'chain_lattice_separator_checks','chain_lattice_separator_terms','chain_pb_rule_checks','chain_pb_cover_assignments',
 'chain_regional_proof_checks','chain_regional_proof_replays','chain_regional_proof_nodes','chain_regional_proof_rows','chain_regional_proof_algebra_ops','chain_regional_proof_unknown')

class ChainError(ValueError):pass


def _values(mask):
    while mask:
        low=mask&-mask;mask^=low;yield low.bit_length()-1


def _mask(raw,n):
    if type(raw)is not str or not raw.startswith('0x') or len(raw)>2+(3*n+3)//4:
        raise ChainError('Malformed chain domain.')
    try:value=int(raw,16)
    except ValueError:raise ChainError('Malformed chain mask.') from None
    if value<0 or value>>(3*n):raise ChainError('Chain mask exceeds physical placements.')
    return value


def _binding(problem):
    return (problem.fingerprint,tuple(tuple(sorted(set(row))) for row in problem.domains),
            tuple(tuple(map(int,row)) for row in problem.masks),tuple(problem.edges),
            tuple(tuple(map(int,row)) for row in problem.neighbors),tuple(tuple(map(int,row)) for row in problem.sides),
            json.dumps(problem.tiles,sort_keys=True,separators=(',',':')))


def _cell(value,n):
    if type(value)is not int or not 0<=value<n:raise ChainError('Invalid chain cell.')
    return value


def _hall(domains,cells,check,stats):
    if type(cells)is not list or not cells or len(cells)>len(domains) or any(type(c)is not int for c in cells) or len(set(cells))!=len(cells):
        raise ChainError('Malformed Hall witness cells.')
    tiles=set()
    for c in cells:
        _cell(c,len(domains));check()
        for code in _values(domains[c]):tiles.add(code//3);stats['chain_hall_placements']+=1
    return len(tiles)<len(cells)


def _loop(problem,domains,placements,target):
    witness=revalidate_witness(problem,placements,target=target)
    return all(domains[c]==1<<code for c,code in witness)


def _regional(problem,domains,data,target,check,stats):
    from constraint_regional_proof_bridge_round26 import verify_reason
    local={};callback_error=[]
    def poll():
        try:check()
        except BaseException as exc:callback_error.append(exc);raise
    try:return verify_reason(problem,domains,data,poll,stats=local,target=target) is True
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        if callback_error:raise
        raise ChainError('Invalid regional reason.') from exc
    finally:
        for key,value in local.items():
            if 'chain_'+key in CHAIN_STAT_KEYS:stats['chain_'+key]=stats.get('chain_'+key,0)+value


def _affine(problem,domains,data,check,stats):
    from constraint_affine_reasons_round26 import verify_affine_conflict
    if set(data)!={'certificate'}:raise ChainError('Malformed affine reason.')
    local={};callback_error=[]
    def poll():
        try:check()
        except BaseException as exc:callback_error.append(exc);raise
    try:return verify_affine_conflict(problem,domains,data['certificate'],poll,local)
    except ValueError as exc:
        if callback_error:raise
        raise ChainError('Invalid affine reason.') from exc
    finally:
        for key,value in local.items():stats['chain_'+key]=stats.get('chain_'+key,0)+value


def _lattice(problem,domains,data,check,stats):
    from constraint_lattice_round26 import verify_separator
    if set(data)!={'certificate'}:raise ChainError('Malformed lattice reason.')
    local={};callback_error=[]
    def poll():
        try:check()
        except BaseException as exc:callback_error.append(exc);raise
    try:return verify_separator(problem,domains,data['certificate'],poll,local)
    except ValueError as exc:
        if callback_error:raise
        raise ChainError('Invalid lattice reason.') from exc
    finally:
        for key,value in local.items():stats['chain_'+key]=stats.get('chain_'+key,0)+value


def _pb(problem,domains,data,check,stats):
    from constraint_pb import primitive_rows,is_integer_row
    from constraint_arithmetic_round26 import replay
    if not {'scope','steps','row_index'}<=set(data)<={'scope','steps','row_index','obstructions'} or type(data['scope'])is not list or not data['scope'] or len(data['scope'])>problem.n:raise ChainError('Malformed PB reason.')
    scope=[];seen=set()
    for item in data['scope']:
        if type(item)is not list or len(item)!=2:raise ChainError('Malformed PB scope.')
        cell=_cell(item[0],problem.n);mask=_mask(item[1],problem.n)
        root=sum(1<<v for v in problem.domains[cell])
        if cell in seen or not mask or mask&~root or domains[cell]&~mask:raise ChainError('PB scope does not contain current domains.')
        seen.add(cell);scope.append((cell,mask))
    if not isinstance(data['steps'],(list,tuple)) or len(data['steps'])>1024 or type(data['row_index'])is not int:raise ChainError('PB proof cap or index.')
    steps=[]
    for step in data['steps']:
        if type(step)is not dict or set(step)!={'parents','rule','args','row'}:raise ChainError('Malformed PB derivation.')
        raw=step['row']
        if not isinstance(raw,(tuple,list)) or len(raw)!=2 or not isinstance(raw[0],(tuple,list)):raise ChainError('Malformed PB result.')
        try:claimed=(tuple(tuple(term) for term in raw[0]),raw[1])
        except TypeError:raise ChainError('Malformed PB terms.') from None
        if not is_integer_row(claimed,max_bits=256):raise ChainError('Noninteger PB result.')
        steps.append({**step,'row':claimed})
    callback_error=[];local={}
    def poll():
        try:check()
        except BaseException as exc:callback_error.append(exc);raise
    axioms=list(primitive_rows(problem,scope))
    if 'obstructions' in data:
        from constraint_pb import ObstructionMiner
        rules=data['obstructions']
        if not isinstance(rules,(list,tuple)) or len(rules)>32:raise ChainError('Obstruction proof bound.')
        miner=ObstructionMiner(problem)
        for record in rules:
            poll()
            if type(record)is not dict or set(record)!={'kind','events','row'}:raise ChainError('Malformed obstruction proof.')
            raw=record['row']
            if not isinstance(raw,(tuple,list)) or len(raw)!=2 or not isinstance(raw[0],(tuple,list)):raise ChainError('Malformed obstruction row.')
            try:cut=(tuple(tuple(term) for term in raw[0]),raw[1])
            except TypeError:raise ChainError('Malformed obstruction term.') from None
            if not is_integer_row(cut,max_bits=256) or not miner.verify({**record,'row':cut}):raise ChainError('Unproved obstruction.')
            axioms.append(cut)
    try:rows=replay(tuple(dict.fromkeys(axioms)),steps,poll,local)
    except (ValueError,TypeError,IndexError) as exc:
        if callback_error:raise
        raise ChainError('PB replay failed.') from exc
    finally:
        stats['chain_pb_rule_checks']=stats.get('chain_pb_rule_checks',0)+local.get('arithmetic_rule_checks',0)
        stats['chain_pb_cover_assignments']=stats.get('chain_pb_cover_assignments',0)+local.get('arithmetic_cover_assignments',0)
    index=data['row_index']
    if not 0<=index<len(rows):raise ChainError('PB conclusion index outside checked proof.')
    grouped={}
    for cell,code,coefficient in rows[index][0]:grouped.setdefault(cell,{})[code]=coefficient
    lower=0
    for cell,coefficients in grouped.items():
        check();lower+=min((coefficients.get(code,0) for code in _values(domains[cell])),default=0)
    return lower>rows[index][1]


def verify_chain(problem,certificate,*,target='gold',check=lambda:None,stats=None):
    """Return a checked original-root mask conflict, or raise ChainError.

    A caller Stop exception propagates unchanged. Failure/unknown never returns
    a conclusion. The checker imposes fixed caps before any proof mutation.
    """
    stats={} if stats is None else stats
    for key in ('chain_events','chain_removed_values','chain_seam_placements','chain_hall_placements','chain_loop_checks'):
        stats.setdefault(key,0)
    check();binding=_binding(problem);n=problem.n
    fields={'format','fingerprint','target','original','root','decisions','events','terminal'}
    if type(certificate)is not dict or set(certificate)!=fields or certificate['format']!=FORMAT:
        raise ChainError('Malformed deduction-chain format.')
    try:size=len(json.dumps(certificate,sort_keys=True,allow_nan=False).encode('utf-8'))
    except (ValueError,TypeError,OverflowError):raise ChainError('Non-JSON chain.') from None
    if size>MAX_BYTES:raise ChainError('Deduction chain byte cap.')
    if target not in ('gold','edges') or certificate['target']!=target or certificate['fingerprint']!=problem.fingerprint:
        raise ChainError('Chain objective or puzzle differs.')
    original=tuple(sum(1<<v for v in row) for row in binding[1])
    if type(certificate['original'])is not list or len(certificate['original'])!=n:
        raise ChainError('Chain original-domain count differs.')
    if tuple(_mask(v,n) for v in certificate['original'])!=original:raise ChainError('Chain original domains differ.')
    roots=certificate['root']
    if type(roots)is not list or len(roots)!=n:raise ChainError('Chain root count differs.')
    root=tuple(_mask(v,n) for v in roots)
    if any(not m or m&~o for m,o in zip(root,original)):raise ChainError('Chain root escapes original input.')
    domains=list(root);decisions=certificate['decisions']
    if type(decisions)is not list or len(decisions)>n:raise ChainError('Malformed decision prefix.')
    seen=set()
    for row in decisions:
        if type(row)is not list or len(row)!=2:raise ChainError('Malformed decision mask.')
        c=_cell(row[0],n);mask=_mask(row[1],n)
        if c in seen or not mask or mask&~root[c]:raise ChainError('Decision escapes lane root.')
        seen.add(c);domains[c]&=mask
    guards=tuple((c,mask) for c,mask in enumerate(domains) if mask!=original[c])
    events=certificate['events']
    if type(events)is not list or len(events)>MAX_EVENTS:raise ChainError('Deduction event cap.')
    for event in events:
        check()
        if type(event)is not dict or set(event)!={'rule','cell','removed','data'}:raise ChainError('Malformed removal event.')
        cell=_cell(event['cell'],n);removed=_mask(event['removed'],n);data=event['data'];rule=event['rule']
        if not removed or removed&~domains[cell] or type(data)is not dict:raise ChainError('Removal is not live or data is malformed.')
        for code in _values(removed):
            check();trial=domains.copy();trial[cell]=1<<code;valid=False
            if rule=='seam':
                if set(data)!={'neighbor'}:raise ChainError('Malformed seam witness.')
                neighbor=_cell(data['neighbor'],n)
                seams=[(side,int(problem.sides[cell,side])) for side in range(3) if int(problem.neighbors[cell,side])==neighbor]
                if not seams:raise ChainError('Seam witness has no shared edge.')
                valid=True
                for other in _values(trial[neighbor]):
                    stats['chain_seam_placements']+=1
                    if all(int(problem.masks[code,s])==reverse_mask(int(problem.masks[other,t])) for s,t in seams):valid=False;break
            elif rule=='boundary':
                if set(data)!={'side'} or type(data['side'])is not int or not 0<=data['side']<3:raise ChainError('Malformed boundary witness.')
                side=data['side'];valid=int(problem.neighbors[cell,side])<0 and int(problem.masks[code,side])!=0
            elif rule=='hall':
                if set(data)!={'cells'}:raise ChainError('Malformed Hall witness.')
                # Conditional assignment reserves this physical tile. Excluding
                # it from every OTHER cell is a puzzle axiom under the trial.
                physical=7<<(3*(code//3))
                for c in range(n):
                    if c!=cell:
                        trial[c]&=~physical
                        seams=[(side,int(problem.sides[cell,side])) for side in range(3) if int(problem.neighbors[cell,side])==c]
                        if seams:
                            allowed=0
                            for other in _values(trial[c]):
                                stats['chain_seam_placements']+=1
                                if all(int(problem.masks[code,s])==reverse_mask(int(problem.masks[other,t])) for s,t in seams):allowed|=1<<other
                            trial[c]=allowed
                valid=_hall(trial,data['cells'],check,stats)
            elif rule=='regional':valid=_regional(problem,trial,data,target,check,stats)
            elif rule=='affine':valid=_affine(problem,trial,data,check,stats)
            elif rule=='pb':valid=_pb(problem,trial,data,check,stats)
            elif rule=='lattice':valid=_lattice(problem,trial,data,check,stats)
            elif rule=='loop':
                if set(data)!={'placements'}:raise ChainError('Malformed Gold witness.')
                stats['chain_loop_checks']+=1
                try:valid=_loop(problem,trial,data['placements'],target)
                except ValueError:valid=False
            else:raise ChainError('Unsupported deduction rule; no consequence admitted.')
            if not valid:raise ChainError('Deduction does not follow from checked prefix.')
            stats['chain_removed_values']+=1
        domains[cell]&=~removed;stats['chain_events']+=1
    check();terminal=certificate['terminal']
    if type(terminal)is not dict or set(terminal)!={'rule','data'} or type(terminal['data'])is not dict:raise ChainError('Missing checked terminal.')
    rule=terminal['rule'];data=terminal['data'];valid=False
    if rule=='empty':
        if set(data)!={'cell'}:raise ChainError('Malformed empty-domain terminal.')
        valid=domains[_cell(data['cell'],n)]==0
    elif rule=='hall':
        if set(data)!={'cells'}:raise ChainError('Malformed terminal Hall witness.')
        valid=_hall(domains,data['cells'],check,stats)
    elif rule=='regional':valid=_regional(problem,domains,data,target,check,stats)
    elif rule=='affine':valid=_affine(problem,domains,data,check,stats)
    elif rule=='pb':valid=_pb(problem,domains,data,check,stats)
    elif rule=='lattice':valid=_lattice(problem,domains,data,check,stats)
    elif rule=='loop':
        if set(data)!={'placements'}:raise ChainError('Malformed terminal Gold witness.')
        stats['chain_loop_checks']+=1
        try:valid=_loop(problem,domains,data['placements'],target)
        except ValueError:valid=False
    else:raise ChainError('Unsupported terminal rule.')
    if not valid:raise ChainError('Trace has no checked contradiction.')
    check()
    if _binding(problem)!=binding:raise ChainError('Puzzle changed while checking deduction chain.')
    return guards


def expand_placement_conflicts(record,*,limit=64):
    """Equivalent bounded singleton clauses; None means expansion unavailable."""
    if type(limit)is not int or limit<1:raise ValueError('Invalid expansion cap.')
    size=1
    for cell,mask in record:
        size*=mask.bit_count()
        if size>limit:return None
    return [tuple((cell,code) for (cell,_),code in zip(record,row))
            for row in itertools.product(*(tuple(_values(mask)) for _,mask in record))]
