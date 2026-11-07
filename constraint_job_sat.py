"""One original-root SAT model per worker, with assumption-scoped job views.

Job restrictions are never posted permanently. Only the owner closes the native
solver; evicting a job view cannot discard another job's learned state.
"""
from constraint_sat_assist import SatAssist


class SharedSatAssistant:
    def __init__(self, problem, target='gold'):
        self.problem = problem
        self.target = target
        self.fingerprint = problem.fingerprint
        self.root = tuple(sum(1 << code for code in set(row)) for row in problem.domains)
        self.helper = SatAssist(problem, self.root, target)
        self.views = 0

    def matches(self, problem, target):
        return (self.fingerprint == problem.fingerprint and self.target == target
                and self.root == tuple(sum(1 << code for code in set(row)) for row in problem.domains))

    def view(self, initial):
        initial = tuple(initial)
        if (len(initial) != len(self.root)
                or any(type(mask) is not int or mask < 0 or mask & ~root
                       for mask, root in zip(initial, self.root))):
            raise ValueError('Shared SAT job domains escape the original root.')
        self.views += 1
        return JobSatAssist(self, initial)

    def close(self):
        self.helper.close()


class JobSatAssist:
    """A job-bound facade; every native job query carries its complete masks."""
    def __init__(self, owner, initial):
        self.owner = owner
        self.initial = initial

    def __getattr__(self, name):
        return getattr(self.owner.helper, name)

    @property
    def witness_pool(self):
        return self.owner.helper.witness_pool

    @witness_pool.setter
    def witness_pool(self, value):
        self.owner.helper.witness_pool = bool(value)

    @property
    def support_cover(self):
        return self.owner.helper.support_cover

    @support_cover.setter
    def support_cover(self, value):
        self.owner.helper.support_cover = bool(value)

    def close(self):
        # The worker owns the shared native solver, not an individual job.
        pass

    def filter_domains(self, domains, check=lambda: None, **kwargs):
        if (len(domains) != len(self.initial)
                or any(type(mask) is not int or mask < 0 or mask & ~root
                       for mask, root in zip(domains, self.initial))):
            raise ValueError('Shared SAT filtering escapes its job domains.')
        return self.owner.helper.filter_domains(domains, check, **kwargs)

    def query(self, placements, check=lambda: None, **kwargs):
        helper = self.owner.helper
        decisions = tuple(sorted(set(tuple(x) for x in placements)))
        if len({c for c, _ in decisions}) != len(decisions):
            raise ValueError('Conflicting SAT placement assumptions.')
        current = list(self.initial)
        chosen = {}
        for cell, code in decisions:
            if (type(cell) is not int or type(code) is not int
                    or not 0 <= cell < len(current) or code < 0
                    or not current[cell] & (1 << code)):
                raise ValueError('SAT assumption outside job domains.')
            current[cell] = 1 << code
            chosen[cell] = code
        if not all(current):
            helper.last_outcome = 'unsat'
            return ()
        literals = tuple((cell, mask) for cell, mask in enumerate(current)
                         if mask != helper.initial[cell])
        core = helper.query_masks(literals, check, **kwargs)
        if core is None:
            return None
        # Root restrictions remain implicit only in this JOB's clause. The
        # shared helper retains the full original-root mask core separately.
        local = []
        for cell, mask in core:
            if not self.initial[cell] & ~mask:
                continue
            if cell not in chosen or not mask & (1 << chosen[cell]):
                raise RuntimeError('SAT core is not implied by the job and decisions.')
            local.append((cell, chosen[cell]))
        return tuple(local)
