"""Project physical signed-inventory hints onto another reference and re-prove them there."""
import json
from pathlib import Path
from time import monotonic


def projections(outside, records, min_fraction=0.75):
    positions = {pair:i for i,pair in enumerate(outside)}
    found = set()
    for record in records:
        try:
            agree = {tuple(row) for row in record['agree']}
            disagree = {tuple(row) for row in record['disagree']}
            if agree.intersection(disagree) or not (agree or disagree):
                continue
            if any(len(p) != 2 or any(type(i) is not int for i in p) for p in agree|disagree):
                continue
            available = (agree|disagree).intersection(positions)
            if len(available) < len(agree|disagree)*min_fraction:
                continue
            mask = sum(1<<positions[p] for p in available)
            value = sum(1<<positions[p] for p in available & disagree)
            found.add((mask,value))
        except (KeyError,TypeError,ValueError):
            continue
    return found


class TransferHints:
    def __init__(self, problem, required, outside, records=None, *, store=None):
        if records is None:
            try:
                rows = json.loads((Path(__file__).parent/'docs/R35_PHYSICAL_LOSS_CORES.json').read_text(encoding='utf8'))
                records = rows['cores'] if rows['fingerprint'] == problem.fingerprint else ()
            except (OSError,ValueError,TypeError,KeyError):
                records = ()
        self.store = store
        records = list(records)
        if store is not None and hasattr(store,'physical'):
            records.extend(store.physical(problem.fingerprint))
        self.todo = projections(outside,records)
        self.problem = problem
        self.required = required
        self.outside = outside
        self.native = None

    def matched(self, value, deadline, work):
        for mask, bits in sorted(self.todo,key=lambda p:(p[0].bit_count(),p)):
            if value & mask != bits:
                continue
            remaining = deadline-monotonic()
            if remaining <= 0:
                raise TimeoutError
            if self.native is None:
                try:
                    from r35_integer_sat import InventorySAT
                    self.native = InventorySAT(self.problem,(),self.required,self.outside,len(self.outside))
                except ImportError:
                    self.todo.clear()
                    return None
            self.todo.remove((mask,bits))
            units = self.native.errors
            variables = {v:i for i,v in enumerate(units)}
            assumptions = [u if bits>>i & 1 else -u for i,u in enumerate(units) if mask>>i & 1]
            remaining = deadline-monotonic()
            if remaining <= 0:
                self.todo.add((mask,bits))
                raise TimeoutError
            result = self.native.decide(min(0.75,remaining),assumptions)
            work['physical_hint_replays'] = work.get('physical_hint_replays',0)+1
            if result is False:
                used = self.native.solver.get_core() or []
                if not set(used) <= set(assumptions):
                    raise ArithmeticError('A physical transfer proof used a foreign assumption.')
                keep = sum(1<<variables[abs(u)] for u in set(used))
                work['physical_hint_proofs'] = work.get('physical_hint_proofs',0)+1
                self.remember(keep,value&keep)
                return keep,value&keep
        return None

    def remember(self, mask, bits):
        if self.store is not None and hasattr(self.store,'remember_physical'):
            self.store.remember_physical(self.problem.fingerprint,self.outside,mask,bits)

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
