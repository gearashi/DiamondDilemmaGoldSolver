"""Shared telemetry and bounds for round26 native exact reasoning."""
SUPPORT_COUNTERS=('support_cache_queries','support_cache_hits','support_cache_misses',
    'support_word_scans','support_cache_stores','support_cache_evictions','support_witness_checks')

def validate_support_policy(reuse,metrics,slots):
    if type(reuse)is not bool or type(metrics)is not bool:
        raise ValueError('Support reuse/metrics must be boolean.')
    if slots is not None and (not reuse or type(slots)is not int or not 8<=slots<=8192 or slots&(slots-1)):
        raise ValueError('Support slots require reuse and a power of two from8 to8192.')
    return bool(reuse or metrics),1024 if slots is None else slots

CLAUSE_COUNTERS=('clause_literal_checks','clause_watch_checks','clause_watch_hits','clause_rescans','clause_wakeups')

MATCHING_COUNTERS=('matching_runs','matching_rows_rebuilt','matching_edge_checks','matching_augmentations',
 'matching_hall_failures','matching_closure_words','matching_tile_tests','matching_exclusions','matching_component_cells')


def validate_matching_state(engine,state):
    """Mechanical validation of trusted local scratch, never a proof importer."""
    import numpy as np
    n,tw=engine.n,engine.reason_words;offsets=engine.offsets
    for lane in range(engine.lanes):
        def values(name,length):return [int(x) for x in state[offsets[name]:offsets[name]+length,lane]]
        match=values('matching_cells',n);owners=values('matching_owners',n)
        parent=values('matching_parent',n);components=values('matching_components',n)
        for rows in (match,owners,parent):
            if any(x>=n and x!=0xffffffff for x in rows):raise ValueError('Invalid matching scratch index.')
        if any(x>=n for x in components):raise ValueError('Invalid matching component index.')
        for cell,tile in enumerate(match):
            if tile!=0xffffffff and owners[tile]!=cell:raise ValueError('Matching owner map differs.')
        for tile,cell in enumerate(owners):
            if cell!=0xffffffff and match[cell]!=tile:raise ValueError('Matching placement map differs.')
        for links,sentinel in ((components,False),(parent,True)):
            for start in range(n):
                seen=set();cell=start
                while cell!=0xffffffff and (sentinel or links[cell]!=cell):
                    if cell in seen:raise ValueError('Cyclic matching scratch links.')
                    seen.add(cell);cell=links[cell]
        meta=values('matching_meta',8)
        if meta[0]>6 or any(meta[i]>n for i in (1,2,3,4,5,7)) or meta[6]>=n or meta[4]>meta[5]:raise ValueError('Invalid matching work cursor.')
        queue=values('matching_queue',meta[5])
        if any(x>=n for x in queue) or len(set(queue))!=len(queue):raise ValueError('Invalid matching work queue.')
        for name,rows in (('matching_dirty',1),('matching_affected',1),('matching_seen',1),('matching_adj',n),('matching_reach',n)):
            words=values(name,rows*tw)
            if n%32 and any(words[row*tw+tw-1]>>(n%32) for row in range(rows)):raise ValueError('Matching bitset has padding bits.')
        phase=int(state[6,lane])
        if phase==5 and meta[0]>=4 or phase==2:
            if 0xffffffff in match or 0xffffffff in owners:raise ValueError('Matching fixed point requires a full bijection.')
            for cell,tile in enumerate(match):
                if not any(engine._mask(state,lane,cell)&(1<<(3*tile+r)) for r in range(3)):raise ValueError('Matching uses an absent placement.')
        if phase==2 and any(values('matching_dirty',tw)):raise ValueError('Matching fixed point retains dirty cells.')


def domain_layout(n,*,joint_support=False,event_propagation=False,precise_reasons=False,reason_slots=None,
                  support_reuse=False,support_metrics=False,support_slots=None,watched_clauses=False,clause_metrics=False,
                  global_matching=False,proof_transport=False,cooperative=False):
    """Authoritative allocation rows, available without importing a GPU library."""
    if type(n)is not int or not 1<=n<=160:raise ValueError('Domain layout requires1..160 cells.')
    if any(type(v)is not bool for v in (joint_support,event_propagation,precise_reasons,watched_clauses,clause_metrics,global_matching,proof_transport,cooperative)):
        raise ValueError('Domain layout policies must be boolean.')
    support_metrics,support_slots=validate_support_policy(support_reuse,support_metrics,support_slots)
    if reason_slots is not None and (not precise_reasons or type(reason_slots)is not int or not 8<=reason_slots<=2048 or reason_slots&(reason_slots-1)):
        raise ValueError('Invalid bounded reason cache.')
    reason_slots=reason_slots if reason_slots is not None else min(2048,max(8,1<<(2*n*(3*n)-1).bit_length()))
    clause_metrics=clause_metrics or watched_clauses;n,w,rw,nc=n,(3*n+31)//32,(n+31)//32,3*n
    fields=[('meta',29),('domain',n*w),('trail',n*nc),('reason',n*rw),
            ('frame_cell',n),('frame_code',n),('frame_remaining',n*w),('frame_trail',n),
            ('frame_union',n),('frame_assign',n),('frame_reason',n*n*rw),('frame_failure',n*rw),
            ('board',n),('owner',n),('assign_log',n),('parent',n*16),('size',n*16),
            ('union_child',n*16),('union_parent',n*16),('union_size',n*16),
            ('clause_length',32),('clause_cell',32*n),('clause_code',32*n)]
    if joint_support:fields.append(('joint_counters',10))
    if event_propagation:fields.extend([('dirty_cells',(n+31)//32),('event_counters',10)])
    if precise_reasons:fields.extend([('precise_root',n*w),('precise_zero',n*w),('precise_count',1),('precise_keys',reason_slots),('precise_why',reason_slots*rw),('precise_counters',16)])
    if support_reuse:fields.extend([('support_keys',support_slots),('support_values',3*support_slots)])
    if support_metrics:fields.append(('support_counters',14))
    if watched_clauses:fields.extend([('watch_positions',64),('watch_cells',n),('watch_pending',1)])
    if clause_metrics:fields.append(('clause_counters',10))
    if global_matching:fields.extend([('matching_cells',n),('matching_owners',n),('matching_components',n),
        ('matching_adj',n*rw),('matching_reach',n*rw),('matching_dirty',rw),('matching_affected',rw),('matching_seen',rw),
        ('matching_parent',n),('matching_queue',n),('matching_meta',8),('matching_counters',18)])
    if proof_transport:fields.append(('proof_meta',3))
    return fields


def memory_estimate(n,lanes=128,*,budget=512*1024*1024,**policy):
    if type(lanes)is not int or lanes<1 or type(budget)is not int or budget<1:raise ValueError('Positive lane/budget values required.')
    fields=domain_layout(n,**policy);rows=sum(size for _,size in fields);nc=3*n;w=(nc+31)//32
    tables=(25*n+nc*33+9*nc*w+(n*((n+31)//32) if policy.get('event_propagation',False) else 0))*4
    return {'bytes_per_lane':4*rows,'state_bytes':4*rows*lanes,'static_table_bytes':tables,
            'minimum_device_payload_bytes':4*rows*lanes+tables,'budget_max_lanes':budget//(4*rows),
            'allocation_budget_bytes':budget,'cooperative':policy.get('cooperative',False)}
