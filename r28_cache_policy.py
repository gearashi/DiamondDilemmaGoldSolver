"""Replay measured cache requests. Values in a trace are immutable per key."""
from collections import OrderedDict
from functools import lru_cache
import sqlite3


def replay(trace, capacity, *, weighted=False, spill=None):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Invalid cache capacity")
    entries = OrderedDict()
    metadata = {}
    used = hits = saved = disk_hits = 0
    age = 0.
    db = sqlite3.connect(spill) if spill else None
    try:
        if db:
            # The spill trial is scoped to this trace. It stores opaque payloads
            # of the observed size and does not treat them as puzzle deductions.
            db.execute("CREATE TEMP TABLE spill (key BLOB PRIMARY KEY, payload BLOB NOT NULL)")
        for key,size,cost in trace:
            if type(key) is not bytes or type(size) is not int or size <= 0 or cost <= 0:
                raise ValueError("Invalid cache trace")
            if metadata.setdefault(key,(size,cost)) != (size,cost):
                raise ValueError("Cost/size changed for immutable trace key")
            if key in entries:
                hits += 1
                saved += cost
                entries.move_to_end(key)
                entries[key] = age + cost/size
                continue
            if db and db.execute('SELECT length(payload) FROM spill WHERE key=?',(key,)).fetchone():
                disk_hits += 1
                saved += cost
            if size > capacity:
                continue
            while used + size > capacity:
                victim = min(entries,key=entries.get) if weighted else next(iter(entries))
                age = entries.pop(victim)
                victim_size = metadata[victim][0]
                used -= victim_size
                if db:
                    db.execute('INSERT OR IGNORE INTO spill VALUES (?,zeroblob(?))',(victim,victim_size))
            entries[key] = age + cost/size
            used += size
        return {'hits':hits,'saved_work':saved,'spill_hits':disk_hits}
    finally:
        if db:
            db.close()


def offline_bound(trace, capacity):
    """Exact look-ahead oracle on small traces, including bypass admission."""
    if len(trace)>100 or len({key for key,_,_ in trace})>14:
        raise ValueError("Small exact trace required")
    sizes={key:size for key,size,_ in trace}
    @lru_cache(None)
    def best(i, resident):
        if i == len(trace):
            return 0
        key, size, cost = trace[i]
        if key in resident:
            return cost + best(i+1, resident)
        answer = best(i+1,resident)
        if size <= capacity:
            for mask in range(1<<len(resident)):
                keep=tuple(v for k,v in enumerate(resident) if mask>>k&1)
                if size+sum(sizes[v] for v in keep)<=capacity:
                    answer=max(answer,best(i+1,tuple(sorted((*keep,key)))))
        return answer
    return best(0,())
