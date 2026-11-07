"""Disjoint symbolic CPU/SAT/CUDA coordination with durable engine cursors.

Only validated completed engine work retires assignments. Scheduling estimates
never prove infeasibility. Engine work counters retain separate units. The
symbolic queue avoids turning every exclusion into another full native job.
"""
from __future__ import annotations
import copy
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field

from constraint_work import WorkScope, checkpoint_boxes, digest, frontier_snapshot
from constraint_symbolic_frontier import SymbolicFrontier, Lease
from constraint_symbolic_transfer import from_checkpoint, portable_snapshot, engine_remaining
from constraint_shared_snapshot import SharedSnapshot
from search_options import normalize_options, validate_workers
from constraint_exchange_round26 import CheckedExchange, import_for_worker
from constraint_context_policy_round26 import ContextPolicy, features, profile_options, PROFILES

SCHEMA = 'diamond-symbolic-coordinator-v1'
BACKENDS = ('cpu', 'sat', 'gpu')
STALLED_REALLOCATION_SLICES = 24
ADAPTIVE_COUNTERS = ('coordinator_stalled_reallocations', 'coordinator_adaptive_cursor_restarts')


def _close_state(state):
    seen = set()
    for value in state.values():
        if id(value) in seen:
            continue
        seen.add(id(value))
        close = getattr(value, 'close', None)
        if callable(close):
            try: close()
            except Exception: pass


def _worker(problem, connection, cancel):
    """One native owner per process; requests contain data, never executable code."""
    state = {}
    current = None
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            identity = (request['ticket'], request['backend'], tuple(request['box']))
            if identity != current:
                _close_state(state)
                state = {}
                current = identity
            import_for_worker(state,problem,request['target'],request['options'],request.get('exchange'))
            mode = request['backend']
            for key,value in request.get('cache_paths',{}).items():state[key]=value
            if request.get('sat_service') is not None and '_sat_service' not in state:state['_sat_service']=request['sat_service']
            started = time.monotonic()
            try:
                def preview(event):
                    connection.send({'kind':'progress','ticket':request['ticket'],'backend':mode,'event':event})
                common = dict(progress=preview,seconds=request['seconds'], seed=request['seed'], target=request['target'],
                              options=profile_options(request['options'], request['profile']) if mode=='cpu' else request['options'], stop=cancel.is_set,
                              resume=request['checkpoint'], learning_state=state)
                if mode == 'cpu':
                    from constraint_dfs import _solve_core
                    result = _solve_core(problem, _initial_domains=request['box'], **common)
                elif mode == 'sat':
                    from constraint_sat_session import solve
                    if common['resume'] is None:
                        scope = WorkScope.for_problem(problem, request['target'], request['box'])
                        common['resume'] = frontier_snapshot(scope, [request['box']])
                    result = solve(problem, **common)
                elif mode == 'gpu':
                    from gpu_domain_runner import solve
                    if common['resume'] is None:
                        scope = WorkScope.for_problem(problem, request['target'], request['box'])
                        common['work_frontier'] = frontier_snapshot(scope, [request['box']])
                    result = solve(problem, replicas=request['replicas'], backend='cuda', **common)
                else:
                    raise ValueError('Unknown exact worker backend.')
                connection.send({'ticket': request['ticket'], 'backend': mode, 'result': result,
                                 'seconds': time.monotonic() - started})
            except BaseException as exc:
                connection.send({'ticket': request['ticket'], 'backend': mode,
                                 'error': type(exc).__name__ + ': ' + str(exc)})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        _close_state(state)
        connection.close()


@dataclass
class Job:
    lease: Lease
    backend: str
    checkpoint: object = None
    metric: int = 0
    slices: int = 0
    unchanged: int = 0
    last_worker: int = -1
    profile: str = 'base'
    prior_stats: dict = field(default_factory=dict)


def _counter_keys(backend):
    """Audited engine counter declarations; everything else is observed maximum.

    Imported modules define numeric event counters. Declared/current/peak gauges
    are excluded explicitly; no opaque future metric is guessed to be a total.
    """
    from functools import lru_cache
    return _counter_registry(backend)


from functools import lru_cache
@lru_cache(maxsize=3)
def _counter_registry(backend):
    import importlib
    counters=set();gauges=set()
    if backend=='cpu':
        module=importlib.import_module('constraint_dfs')
        for name,value in vars(module).items():
            if isinstance(value,(tuple,list,set,frozenset)) and all(isinstance(v,str) for v in value):
                if name.endswith(('_STAT_KEYS','_STATS')):counters.update(value)
                if 'GAUGE' in name:gauges.update(value)
        gauges.update(('conflict_cache_entries','loop_witnesses','native_search_peak_trail','native_search_peak_depth'))
    elif backend=='sat':
        counters.update(('session_model_builds','session_warm_slices','session_native_calls','session_cubes_closed','session_gold_cuts','session_decisions','session_candidates','session_imported_loops','session_imported_masks','session_total_decisions',
          'session_core_candidates','session_core_verifications','session_core_model_builds','session_core_decisions','session_core_verified','session_core_unknown','session_core_boxes_removed','session_core_assignments_removed'))
        for name in ('constraint_sat_persistence','constraint_sat_guards_round26','constraint_sat_simplify_round26','constraint_sat_theory_round26'):
            module=importlib.import_module(name);counters.update(module.STAT_KEYS);gauges.update(getattr(module,'GAUGES',()))
            counters.update(getattr(module,'EXTERNAL_KEYS',()))
        counters.update('regional_dag_'+suffix for suffix in ('regions','arcs','constraints','compile_transitions','compile_states','cap_skips','expanded_histories','compile_reuses','compile_resumes','cursor_loads','cursor_saves','cursor_io_failures','compile_evictions','encoder_replays'))
    elif backend=='gpu':
        for name in ('gpu_domain_runner','gpu_lane_pool_round25','gpu_shared_round25'):
            module=importlib.import_module(name);counters.update(module.COUNTERS);gauges.update(getattr(module,'GAUGES',()))
    else:raise ValueError('Unknown counter backend.')
    return frozenset(counters-gauges)


def _numeric_stats(payload):
    if not isinstance(payload,dict) or len(payload)>8192:raise ValueError('Invalid worker telemetry dictionary.')
    result={}
    for name,value in payload.items():
        if not isinstance(name,str) or not 1<=len(name)<=160:raise ValueError('Invalid worker metric name.')
        if type(value) in (int,float):
            if value<0 or type(value)is float and not math.isfinite(value):raise ValueError('Invalid nonnegative worker metric.')
            result[name]=value
    return result


def _restore_engine_stats(payload):
    if not isinstance(payload,dict) or set(payload)!=set(BACKENDS):raise ValueError('Invalid engine telemetry namespaces.')
    for backend,metrics in payload.items():
        if not isinstance(metrics,dict) or len(metrics)>8192:raise ValueError('Invalid engine telemetry count.')
        for name,row in metrics.items():
            _numeric_stats({name:0})
            if not isinstance(row,dict) or set(row)!={'mode','value','maximum','resets'} or row['mode'] not in ('monotonic_delta','maximum_observed') or type(row['resets'])is not int or row['resets']<0:raise ValueError('Invalid engine metric semantics.')
            if any(type(row[k])not in (int,float) or row[k]<0 or type(row[k])is float and not math.isfinite(row[k]) for k in ('value','maximum')):raise ValueError('Invalid cumulative engine metric.')
            if row['mode']=='monotonic_delta' and (name not in _counter_keys(backend) or type(row['value'])is not int or type(row['maximum'])is not int):raise ValueError('Undeclared cumulative engine metric.')
            if row['mode']=='monotonic_delta' and row['value']<row['maximum']:raise ValueError('Counter total is below observed value.')
            if row['mode']=='maximum_observed' and row['value']!=row['maximum']:raise ValueError('Observed maximum differs from value.')
    return copy.deepcopy(payload)


class AllocationPolicy:
    """Bounded context observations affect order/lease length only.

    Different engines' decision counters are never numerically ranked. Selection
    uses conclusive jobs and observed elapsed cost, with forced exploration and
    periodic migration; work-unit comparisons remain in the evaluation report.
    """
    def __init__(self, payload=None):
        self.records = {}
        self.turn = 0
        if payload is not None:
            if not isinstance(payload, dict) or set(payload) != {'turn', 'records'} or type(payload['turn']) is not int or payload['turn'] < 0:
                raise ValueError('Invalid allocation policy state.')
            if not isinstance(payload['records'], dict) or len(payload['records']) > 96:
                raise ValueError('Invalid allocation contexts.')
            for key, row in payload['records'].items():
                if not isinstance(key, str) or not isinstance(row, list) or len(row) != 3:
                    raise ValueError('Invalid allocation observation.')
                calls, completed, seconds = row
                if (type(calls) is not int or type(completed) is not int or not 0 <= completed <= calls
                        or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError('Invalid allocation counters.')
            self.records = copy.deepcopy(payload['records'])
            self.turn = payload['turn']

    @staticmethod
    def context(box):
        widths = [x.bit_count() for x in box]
        free = sum(x > 1 for x in widths)
        return (min(3, free // 40), min(3, max(widths).bit_length() // 3))

    def key(self, box, backend):
        return ':'.join(map(str, (*self.context(box), backend)))

    def choose(self, box, available, preferred):
        self.turn += 1
        if len(available) == 1:
            return available[0]
        rows = {mode:self.records.get(self.key(box, mode), [0, 0, 0.0]) for mode in available}
        if self.turn % 4 == 0:
            return min(available, key=lambda mode: (rows[mode][0], mode))
        unseen = [mode for mode in available if rows[mode][0] == 0]
        if unseen:
            return preferred if preferred in unseen else unseen[0]
        return max(available, key=lambda mode: ((rows[mode][1] + 1) / (rows[mode][2] + 1), mode == preferred))

    def observe(self, box, backend, elapsed, complete):
        key = self.key(box, backend)
        row = self.records.setdefault(key, [0, 0, 0.0])
        row[0] += 1
        row[1] += int(complete)
        row[2] += elapsed

    def snapshot(self):
        return {'turn': self.turn, 'records': copy.deepcopy(self.records)}


def split_box(box):
    """A complete binary partition; select one side for the next worker."""
    candidates = [i for i, mask in enumerate(box) if mask.bit_count() > 1]
    if not candidates:
        return box
    axis = min(candidates, key=lambda i: (box[i].bit_count(), i))
    rest = box[axis]
    selected = 0
    for _ in range(rest.bit_count() // 2):
        bit = rest & -rest
        selected |= bit
        rest ^= bit
    result = list(box)
    result[axis] = selected
    return tuple(result)


class Coordinator:
    def __init__(self, problem, *, target='gold', seed=0, options=None, resume=None,
                 preferred='cpu', workers=1, replicas=128):
        if target not in ('gold', 'edges') or type(seed) is not int or preferred not in BACKENDS:
            raise ValueError('Invalid coordinated search configuration.')
        self.problem, self.target, self.seed = problem, target, seed
        self.options = normalize_options(resume.get('options') if options is None and isinstance(resume, dict) and resume.get('schema') == SCHEMA else options)
        self.workers = validate_workers(workers)
        if type(replicas) is not int or not 1 <= replicas <= 512:
            raise ValueError('Coordinated CUDA domain search supports 1..512 lanes; use128 initially.')
        self.replicas, self.preferred = replicas, preferred
        self.jobs = {}
        self.policy = AllocationPolicy()
        self.context_policy = ContextPolicy()
        self.stats = dict.fromkeys(('coordinator_slices', 'coordinator_jobs_closed', 'coordinator_transfers',
                                    'coordinator_replayed_slices', 'coordinator_cpu_nodes', 'coordinator_sat_decisions',
                                    'coordinator_gpu_decisions', 'coordinator_worker_starts', 'coordinator_adaptive_splits', *ADAPTIVE_COUNTERS), 0)
        self.engine_stats={backend:{} for backend in BACKENDS}
        self.engine_stats_complete=True
        self.elapsed = 0.0
        self.origin = {}
        self.cache_paths={};self.sat_service=None
        self.exchange=CheckedExchange(problem,target,self.options)
        self.context = mp.get_context('spawn')
        self.slots = []
        self.result_codes = None
        self.result_validation = None
        self.error = None
        self._preview=None
        if isinstance(resume, dict) and resume.get('schema') == SCHEMA:
            self._restore(resume)
        elif resume is not None:
            self.frontier = from_checkpoint(problem, resume, target=target, options=self.options)
            self.origin = {'kind': 'imported_remaining_work', 'complete_original_root': False}
        else:
            from constraint_symmetry import canonical_root_domains
            root, symmetry = canonical_root_domains(problem)
            scope = WorkScope.for_problem(problem, target, tuple(root))
            self.frontier = SymbolicFrontier(scope)
            self.origin = {'kind': 'fresh_search', 'symmetry': symmetry, 'complete_original_root': True}
        self.available = [preferred]
        if self.options['coordinated_sat'] and 'sat' not in self.available:
            self.available.append('sat')
        if self.options['coordinated_gpu'] and 'gpu' not in self.available:
            self.available.append('gpu')
        if self.options['adaptive_allocation'] and 'cpu' not in self.available:
            self.available.append('cpu')
        for ticket, job in list(self.jobs.items()):
            if job.backend not in self.available:
                self.frontier.reclaim(job.lease)
                del self.jobs[ticket]
                self.stats['coordinator_transfers'] += 1

    def _restore(self, payload):
        required = {'schema', 'frontier', 'jobs', 'states', 'seed', 'target', 'options',
                    'stats', 'elapsed', 'origin', 'policy', 'context_policy', 'terminal', 'exchange', 'digest'}
        optional={'engine_stats','engine_stats_complete'}
        if not required<=set(payload) or set(payload)-required not in (set(),optional) or digest({k:v for k,v in payload.items() if k != 'digest'}) != payload['digest']:
            raise ValueError('Coordinated checkpoint integrity differs.')
        if payload['target'] != self.target or type(payload['seed']) is not int:
            raise ValueError('Coordinated objective or seed differs.')
        self.seed = payload['seed']
        normalize_options(payload['options'])
        self.frontier = SymbolicFrontier.restore(payload['frontier'], WorkScope.for_problem(self.problem, self.target))
        if not isinstance(payload['jobs'], list) or len(payload['jobs']) != len(self.frontier.leases):
            raise ValueError('Coordinated jobs and leases differ.')
        states = SharedSnapshot.restore(payload['states'])
        if len(states.roots) != len(payload['jobs']):
            raise ValueError('Coordinated native state count differs.')
        for index, row in enumerate(payload['jobs']):
            if not isinstance(row, dict) or set(row) not in ({'ticket', 'backend', 'metric', 'slices', 'unchanged', 'profile'},{'ticket', 'backend', 'metric', 'slices', 'unchanged', 'profile','prior_stats'}):
                raise ValueError('Invalid coordinated job.')
            ticket = row['ticket']
            if type(ticket) is not int or ticket not in self.frontier.leases or ticket in self.jobs or row['backend'] not in BACKENDS:
                raise ValueError('Unknown coordinated lease/backend.')
            if any(type(row[key]) is not int or row[key] < 0 for key in ('metric', 'slices', 'unchanged')):
                raise ValueError('Malformed coordinated counters.')
            lease, owned = self.frontier.leases[ticket]
            if row['profile'] not in PROFILES:raise ValueError('Invalid coordinated search profile.')
            native = states.get(index)
            if native is not None:
                _, rows = engine_remaining(self.problem, row['backend'], native, target=self.target, options=self.options)
                node = self.frontier.sets.union_boxes(rows)
                if node != owned:
                    raise ValueError('Native cursor and symbolic lease cover differ.')
            elif owned != self.frontier.sets.box(lease.box):
                raise ValueError('Missing cursor for a partially consumed lease.')
            self.jobs[ticket] = Job(lease, row['backend'], native, row['metric'], row['slices'], row['unchanged'], profile=row['profile'],prior_stats=_numeric_stats(row.get('prior_stats',{})))
        if (not isinstance(payload['stats'], dict)
                or set(payload['stats']) not in (set(self.stats), set(self.stats)-set(ADAPTIVE_COUNTERS))
                or any(type(x) is not int or x < 0 for x in payload['stats'].values())):
            raise ValueError('Invalid coordinated statistics.')
        self.stats.update(payload['stats'])
        if 'engine_stats' in payload:
            if type(payload['engine_stats_complete'])is not bool:raise ValueError('Invalid engine telemetry coverage flag.')
            self.engine_stats=_restore_engine_stats(payload['engine_stats']);self.engine_stats_complete=payload['engine_stats_complete']
        else:self.engine_stats_complete=False
        if type(payload['elapsed']) not in (int, float) or not math.isfinite(payload['elapsed']) or payload['elapsed'] < 0:
            raise ValueError('Invalid coordinated elapsed time.')
        self.elapsed = payload['elapsed']
        if not isinstance(payload['origin'], dict):
            raise ValueError('Invalid coordinated origin metadata.')
        self.origin = copy.deepcopy(payload['origin'])
        self.exchange.offer(payload['exchange'])
        self.policy = AllocationPolicy(payload['policy'])
        self.context_policy = ContextPolicy(payload['context_policy'])
        terminal = payload['terminal']
        if terminal is not None:
            self._validate_solution(terminal)
            self.result_codes = list(terminal)
            self.result_validation = self.problem.validate(terminal)

    def snapshot(self):
        states = SharedSnapshot()
        jobs = []
        for ticket, job in sorted(self.jobs.items()):
            states.add(job.checkpoint)
            jobs.append({'ticket':ticket, 'backend':job.backend, 'metric':job.metric,
                         'slices':job.slices, 'unchanged':job.unchanged, 'profile':job.profile,'prior_stats':dict(job.prior_stats)})
        body = {'schema':SCHEMA, 'frontier':self.frontier.snapshot(), 'jobs':jobs,
                'states':states.snapshot(), 'seed':self.seed, 'target':self.target,
                'options':dict(self.options), 'stats':dict(self.stats), 'elapsed':self.elapsed,
                'origin':copy.deepcopy(self.origin), 'policy':self.policy.snapshot(),
                'terminal':self.result_codes, 'context_policy':self.context_policy.snapshot(), 'exchange':self.exchange.snapshot(pending=True),
                'engine_stats':copy.deepcopy(self.engine_stats),'engine_stats_complete':self.engine_stats_complete}
        body['digest'] = digest(body)
        return body

    def _start_slot(self, index):
        parent, child = self.context.Pipe()
        cancel = self.context.Event()
        process = self.context.Process(target=_worker, args=(self.problem, child, cancel), daemon=True)
        process.start()
        child.close()
        slot = {'process':process, 'connection':parent, 'cancel':cancel, 'ticket':None, 'index':index}
        self.stats['coordinator_worker_starts'] += 1
        return slot

    def _slots_needed(self):
        cpu_slots = self.workers if self.preferred == 'cpu' or self.options['adaptive_allocation'] else 1
        return min(65, cpu_slots + int('gpu' in self.available and len(self.available) > 1))

    def _busy(self):
        return {slot['ticket'] for slot in self.slots if slot['ticket'] is not None}

    def _dispatch(self, slot, seconds):
        busy = self._busy()
        gpu_busy = any(self.jobs[t].backend == 'gpu' for t in busy)
        cpu_busy = sum(self.jobs[t].backend != 'gpu' for t in busy)
        def eligible(mode):
            return not gpu_busy if mode == 'gpu' else cpu_busy < self.workers
        queued = [job for ticket, job in self.jobs.items() if ticket not in busy and eligible(job.backend)]
        if queued:
            job = min(queued, key=lambda j:(j.last_worker != slot['index'], j.slices, j.lease.ticket))
        else:
            if not self.frontier.pending:
                return False
            box = self.frontier.sets.first_box(self.frontier.pending)
            available = [mode for mode in self.available if eligible(mode)]
            if not available:
                return False
            mode = self.policy.choose(box, available, self.preferred) if self.options['adaptive_allocation'] else available[slot['index'] % len(available)]
            # Partition only when another idle worker needs work, or a stalled
            # region has explicitly been returned for reallocation.
            split = self._slots_needed() > 1 or self.options['adaptive_allocation']
            lease = self.frontier.claim('coordinator', choose=split_box if split else None)
            profile = self.context_policy.choose(features(self.problem, lease.box)) if mode=='cpu' and self.options['adaptive_allocation'] else 'base'
            job = Job(lease, mode, profile=profile)
            self.jobs[lease.ticket] = job
            if split and lease.box != box:
                self.stats['coordinator_adaptive_splits'] += 1
        slot['cancel'].clear()
        allowance = min(seconds, min(30.0, 1.0 * (2 ** min(job.unchanged, 5))))
        if allowance <= 0:
            return False
        slot['connection'].send({'ticket':job.lease.ticket, 'backend':job.backend, 'box':job.lease.box,
                                 'seconds':allowance, 'seed':self.seed, 'target':self.target,
                                 'options':self.options, 'replicas':self.replicas, 'checkpoint':job.checkpoint, 'profile':job.profile, 'cache_paths':self.cache_paths,
                                 'exchange':self.exchange.snapshot(),
                                 'sat_service':self.sat_service.client(job.lease.box) if self.sat_service is not None and job.backend=='cpu' else None})
        slot['ticket'] = job.lease.ticket
        job.last_worker = slot['index']
        return True

    def _validate_solution(self, codes, box=None):
        if (not isinstance(codes, list) or len(codes) != self.problem.n
                or any(type(code) is not int or code < 0 or code >= 3*self.problem.n for code in codes)):
            raise ValueError('Malformed worker solution.')
        allowed = self.frontier.scope.root if box is None else box
        if any(not allowed[cell] >> code & 1 for cell, code in enumerate(codes)):
            raise ValueError('Worker solution escapes its owned root.')
        report = self.problem.validate(codes)
        if (not report['unique_tiles'] or report['matched_edges'] != report['total_edges']
                or self.target == 'gold' and not report['valid']):
            raise ValueError('Worker solution failed independent validation.')
        return report

    def _stage_engine_stats(self,job,payload):
        numeric=_numeric_stats(payload);updated=copy.deepcopy(self.engine_stats[job.backend]);known=_counter_keys(job.backend)
        for name,value in numeric.items():
            row=updated.get(name)
            if row is None:
                row=updated[name]={'mode':'monotonic_delta' if name in known and type(value)is int else 'maximum_observed','value':0,'maximum':0,'resets':0}
            row['maximum']=max(row['maximum'],value)
            previous=job.prior_stats.get(name,0)
            if row['mode']=='monotonic_delta':
                if type(value)is not int or value<previous:
                    # A reset/gauge-like stream is no longer a defensible work
                    # total. Preserve the explicitly labelled observed maximum.
                    row['mode']='maximum_observed';row['resets']+=1;row['value']=row['maximum']
                else:row['value']+=value-previous
            else:row['value']=row['maximum']
        prior=dict(job.prior_stats);prior.update(numeric)
        return updated,prior

    def _reallocate_stalled(self, job):
        """Move one exact unfinished cube to a different engine or CPU profile.

        Native cursor work inside that cube can be repeated; completed ownership
        is never restored. The rest stays symbolic rather than becoming jobs.
        All fallible set construction precedes the atomic ownership replacement.
        """
        if job.unchanged < STALLED_REALLOCATION_SLICES:
            return False
        owned = self.frontier.leases[job.lease.ticket][1]
        box = self.frontier.sets.first_box(owned)
        alternatives = [mode for mode in self.available if mode != job.backend]
        if alternatives:
            mode = self.policy.choose(box, alternatives, alternatives[0])
            profile = self.context_policy.choose(features(self.problem, box)) if mode == 'cpu' else 'base'
        elif job.backend == 'cpu' and any(mask.bit_count() > 1 for mask in box):
            current = profile_options(self.options, job.profile)
            candidates = [PROFILES[(PROFILES.index(job.profile) + offset) % len(PROFILES)]
                          for offset in range(1, len(PROFILES))]
            profile = next((item for item in candidates if profile_options(self.options, item) != current), None)
            if profile is None:
                return False
            mode = 'cpu'
        else:
            return False
        selected = self.frontier.sets.box(box)
        remainder = self.frontier.sets.difference(owned, selected)
        pending = self.frontier.sets.union(self.frontier.pending, remainder)
        lease = Lease(self.frontier.next_ticket, 'coordinator', box)
        replacement = Job(lease, mode, profile=profile)
        # This is the same checked ownership operation as reclaim + claim, but
        # selects this stalled job's cube, not an unrelated first pending cube.
        self.frontier.pending = pending
        del self.frontier.leases[job.lease.ticket]
        self.frontier.leases[lease.ticket] = (lease, selected)
        self.frontier.next_ticket += 1
        self.frontier.stats['symbolic_requeues'] += 1
        self.frontier.stats['symbolic_leases'] += 1
        del self.jobs[job.lease.ticket]
        self.jobs[lease.ticket] = replacement
        self.stats['coordinator_transfers'] += 1
        self.stats['coordinator_stalled_reallocations'] += 1
        self.stats['coordinator_adaptive_cursor_restarts'] += int(job.checkpoint is not None)
        return True

    def _receive(self, slot):
        message = slot['connection'].recv()
        ticket = slot['ticket']
        if not isinstance(message, dict) or message.get('ticket') != ticket or ticket not in self.jobs:
            raise ValueError('Worker response has no matching lease.')
        job = self.jobs[ticket]
        if message.get('backend') != job.backend:
            raise ValueError('Worker backend changed inside a lease.')
        if message.get('kind')=='progress':
            event=message.get('event')
            if not isinstance(event,dict):raise ValueError('Malformed worker progress.')
            codes=event.get('codes')
            if codes is not None and (not isinstance(codes,list) or len(codes)!=self.problem.n or any(type(code)is not int or code < -1 or code>=3*self.problem.n or code>=0 and not job.lease.box[cell]>>code&1 for cell,code in enumerate(codes))):
                raise ValueError('Worker preview escaped its lease.')
            self._preview={'codes':codes,'is_partial':event.get('is_partial',True),
                           'assigned_tiles':sum(code>=0 for code in codes) if codes is not None else 0,
                           'preview_backend':job.backend,'preview_native_nodes':event.get('nodes',0)}
            return
        if 'error' in message:raise RuntimeError(message['error'])
        result = message['result']
        status = result.get('status')
        metric = result.get('nodes')
        if type(metric) is not int or metric < job.metric:
            raise ValueError('Worker work counters went backwards.')
        delta = metric - job.metric
        engine_stats,prior_stats=self._stage_engine_stats(job,result.get('stats',{}))
        can_reallocate=False
        if status in ('solved', 'edge_perfect'):
            codes = result.get('codes')
            report = self._validate_solution(codes, job.lease.box)
            self.result_codes, self.result_validation = codes, report
        elif status == 'infeasible' and result.get('complete') is True:
            self.frontier.commit(job.lease, ())
            self.stats['coordinator_jobs_closed'] += 1
            del self.jobs[ticket]
        elif status in ('timeout', 'stopped') and result.get('complete') is False:
            native = result.get('checkpoint')
            _, rows = engine_remaining(self.problem, job.backend, native, target=self.target, options=self.options)
            old_owned = self.frontier.leases[ticket][1]
            new_owned = self.frontier.sets.union_boxes(rows)
            self.frontier.commit(job.lease, rows, release=False)
            job.checkpoint = native
            job.metric = metric
            job.unchanged = job.unchanged + 1 if old_owned == new_owned else 0
            job.slices += 1
            if ticket not in self.frontier.leases:
                del self.jobs[ticket]
                self.stats['coordinator_jobs_closed'] += 1
            elif self.options['adaptive_allocation']:
                can_reallocate=True
        else:
            raise ValueError('Unrecognized/inconsistent exact worker result.')
        self.engine_stats[job.backend]=engine_stats;job.prior_stats=prior_stats
        slot['ticket'] = None
        # Commit accepted-slice accounting together with its durable cursor,
        # before optional set allocation, proof exchange or policy updates.
        self.stats['coordinator_' + {'cpu':'cpu_nodes','sat':'sat_decisions','gpu':'gpu_decisions'}[job.backend]] += delta
        self.stats['coordinator_slices'] += 1
        if can_reallocate:
            from constraint_symbolic_work import DiagramLimit
            try:
                if not self._reallocate_stalled(job) and job.slices % 8 == 0 and job.unchanged == 0:
                    self.frontier.reclaim(job.lease)
                    del self.jobs[ticket]
                    self.stats['coordinator_transfers'] += 1
                    self.stats['coordinator_adaptive_cursor_restarts'] += int(job.checkpoint is not None)
            except DiagramLimit:
                pass  # Optional reassignment capped; keep the accepted cursor.
        self.exchange.offer(result)
        self.exchange.advance()
        self.exchange.retire_verified(self.frontier)
        if self.stats['coordinator_slices'] % 64 == 0:self.frontier.optimize(steps=2000)
        complete=status in ('infeasible','solved','edge_perfect')
        self.policy.observe(job.lease.box, job.backend, message['seconds'], complete)
        if job.backend=='cpu':
            self.context_policy.observe(features(self.problem,job.lease.box),job.profile,nodes=metric,complete=complete,case=str(job.lease.ticket))

    def drain(self):
        """Checkpoint active leases while preserving idle worker model caches."""
        for slot in self.slots:
            slot['cancel'].set()
        deadline = time.monotonic() + 10
        healthy = []
        for slot in self.slots:
            if slot['ticket'] is not None:
                while slot['process'].is_alive() and time.monotonic() < deadline:
                    if slot['connection'].poll(.05):
                        try: self._receive(slot)
                        except Exception as exc:
                            self.error = self.error or str(exc)
                            if slot['process'].is_alive():slot['process'].terminate()
                            break
                        if slot['ticket'] is None:break
            if slot['ticket'] is not None or not slot['process'].is_alive():
                if slot['ticket'] is not None:
                    self.stats['coordinator_replayed_slices'] += 1
                if slot['process'].is_alive(): slot['process'].terminate()
                slot['process'].join(timeout=2)
                slot['connection'].close()
            else:
                healthy.append(slot)
        self.slots = healthy

    def close(self):
        self.drain()
        for slot in self.slots:
            try: slot['connection'].send(None)
            except (BrokenPipeError, EOFError, OSError): pass
            slot['process'].join(timeout=1)
            if slot['process'].is_alive():
                slot['process'].terminate()
                slot['process'].join(timeout=2)
            slot['connection'].close()
        self.slots = []
        if self.sat_service is not None:self.sat_service.close();self.sat_service=None

    def dispose(self):
        self.close();self.exchange.close()

    def advance(self, seconds, *, stop=None, progress=None, keep_alive=False):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Coordinated duration must be finite and nonnegative.')
        stop = stop or (lambda:False)
        progress = progress or (lambda event:None)
        start = time.monotonic()
        deadline = start + seconds
        last_report = start
        last_checkpoint = start
        status = 'timeout'
        self.error = None
        try:
            if self.result_codes is not None:
                status = 'solved' if self.result_validation['valid'] else 'edge_perfect'
            elif stop():
                status = 'stopped'
            elif not self.frontier.pending and not self.jobs:
                status = 'infeasible'
            elif seconds:
                if self.options['shared_sat_service'] and self.sat_service is None:
                    from constraint_sat_service_round26 import SATServiceOwner
                    self.sat_service=SATServiceOwner(self.problem,self.target,directory=self.cache_paths.get('_reasoning_cache_dir'),instances=self.options['sat_service_instances'])
                while len(self.slots) < self._slots_needed():
                    index = min(set(range(self._slots_needed())) - {slot['index'] for slot in self.slots})
                    self.slots.append(self._start_slot(index))
                while True:
                    if self.result_codes is not None:
                        status = 'solved' if self.result_validation['valid'] else 'edge_perfect'
                        break
                    if not self.frontier.pending and not self.jobs:
                        status = 'infeasible'
                        break
                    if stop() or time.monotonic() >= deadline:
                        status = 'stopped' if stop() else 'timeout'
                        break
                    for slot in self.slots:
                        if slot['ticket'] is not None:
                            if slot['connection'].poll():
                                try:self._receive(slot)
                                except Exception:
                                    if slot['process'].is_alive():slot['process'].terminate()
                                    raise
                            elif not slot['process'].is_alive():
                                raise RuntimeError('Owned worker exited; the last committed cursor is preserved.')
                        if slot['ticket'] is None and self.result_codes is None:
                            self._dispatch(slot, max(0, deadline - time.monotonic()))
                    now = time.monotonic()
                    if now - last_report >= 1:
                        event={'event':'progress','nodes':self.stats['coordinator_slices'],
                               'stats':self.public_stats(),'node_metric':'committed engine slices; native work units separate'}
                        if self._preview is not None:event.update(self._preview)
                        if now-last_checkpoint>=5:
                            event['checkpoint']=self.snapshot();last_checkpoint=now
                        progress(event)
                        last_report = now
                    time.sleep(.01)
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            status = 'error'
        finally:
            # Cooperative cancellation commits responsive workers before saving.
            # Native state survives within each lease checkpoint. Hard failure
            # can replay only the uncommitted slice, explicitly counted.
            self.drain()
            if not keep_alive or status != 'timeout' or self.error or self.result_codes is not None:
                self.close()
            self.elapsed += time.monotonic() - start
        if self.result_codes is not None:
            status = 'solved' if self.result_validation['valid'] else 'edge_perfect'
        elif self.error:
            status = 'error'
        elif not self.frontier.pending and not self.jobs:
            status = 'infeasible'
        return {'status':status, 'complete':status in ('solved','edge_perfect','infeasible'),
                'codes':self.result_codes,'validation':self.result_validation,
                'nodes':self.stats['coordinator_slices'],'stats':self.public_stats(),
                'checkpoint':self.snapshot(),'elapsed_seconds':self.elapsed,'error':self.error,
                'node_metric':'committed engine slices; CPU/SAT/GPU decisions remain separate'}

    def public_stats(self):
        engine={f'coordinator_{backend}_{row["mode"]}_{name}':row['value'] for backend,metrics in self.engine_stats.items() for name,row in metrics.items()}
        resets=sum(row['resets'] for metrics in self.engine_stats.values() for row in metrics.values())
        return {**engine,'coordinator_engine_metric_resets':resets,'coordinator_engine_stats_complete':self.engine_stats_complete,**self.stats, **self.frontier.stats, **self.exchange.stats, 'coordinator_pending_nodes':len(self.frontier.sets.graph.nodes),
                'coordinator_live_leases':len(self.jobs),'coordinator_cpu_workers':self.workers,
                'coordinator_worker_processes':len(self.slots),
                'coordinator_raw_remaining':str(self.frontier.remaining_volume())}


def solve(problem, *, seconds=10, seed=0, target='gold', options=None, resume=None,
          workers=1, replicas=128, backend='auto', learning_state=None, stop=None,
          progress=None, hints=None, engine_selection='dfs'):
    if any(not row for row in problem.domains):
        return {'status':'infeasible','complete':True,'nodes':0,'stats':{},'checkpoint':None}
    preferred = {'dfs':'cpu','parallel-dfs':'cpu','sat':'sat','gpu-domains':'gpu'}.get(engine_selection)
    if preferred is None:
        raise ValueError('Symbolic coordination requires an exact resumable engine.')
    if preferred == 'gpu' and backend == 'webgpu':
        raise ValueError('The exact domain engine requires CUDA.')
    requested_options=options
    options = normalize_options(resume.get('options') if options is None and isinstance(resume, dict) and resume.get('schema') == SCHEMA else options)
    if learning_state is not None and not isinstance(learning_state, dict):
        raise ValueError('Coordinated learning state must be a trusted dictionary.')
    coordinator = learning_state.get('_symbolic_coordinator') if learning_state is not None else None
    if coordinator is not None:
        if not isinstance(coordinator, Coordinator) or not WorkScope.for_problem(problem, target).accepts(coordinator.frontier.scope):
            raise ValueError('Warm coordinator scope differs.')
        if requested_options is None and resume is None:options=dict(coordinator.options)
        if preferred != 'gpu' and not options['coordinated_gpu']:replicas=128
        unchanged = (coordinator.options == options and coordinator.workers == workers
                     and coordinator.preferred == preferred and coordinator.replicas == replicas
                     and (resume is None or digest(coordinator.snapshot()) == digest(resume)))
        if not unchanged:
            coordinator.dispose()
            if resume is None:resume=coordinator.snapshot()
            coordinator = None
    if coordinator is None:
        if preferred != 'gpu' and not options['coordinated_gpu']:replicas=128
        coordinator = Coordinator(problem, target=target, seed=seed, options=options, resume=resume,
                                  preferred=preferred, workers=workers, replicas=replicas)
        if learning_state is not None:
            learning_state['_symbolic_coordinator'] = coordinator
    if backend=='webgpu' and (preferred=='gpu' or options['coordinated_gpu']):
        raise ValueError('Coordinated exact GPU work requires CUDA.')
    if learning_state is not None:
        coordinator.cache_paths={key:learning_state[key] for key in ('_reasoning_cache_dir','_regional_cache_dir') if key in learning_state}
    try:
        return coordinator.advance(seconds, stop=stop, progress=progress, keep_alive=learning_state is not None)
    finally:
        if learning_state is None:
            coordinator.dispose()
