"""Original Gold pose clauses for explicitly selected exact-search frontiers."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _trigger(seed):
    from r35_pose_departures import reference_images
    images = reference_images(seed)
    return frozenset((site,bit) for image in images for site,bit in image)


class FrontierPoses:
    """The active private solver invocation owns its validated puzzle and domains."""
    def __init__(self, problem):
        self.problem = problem
        self.catalogue = None
        self.checked = False
        self.last_signature = None
        self.last_actions = None

    def prune(self, domains):
        if self.problem.n != 160 or not any(d and d & (d-1) == 0 for d in domains):
            return domains, None
        if not self.checked:
            from r35_claim_catalogue import published_scope
            from r35_repair_seeds import active_catalog
            self.checked = True
            if published_scope(self.problem):
                record = active_catalog(self.problem)
                if record and record.get('lower_bound') == 95:
                    from r35_pose_departures import indexed_rows, reference_images
                    seed = tuple(record['codes'])
                    rows = (*record['rows'], *record.get('post_master',()))
                    self.catalogue = (_trigger(seed), reference_images(seed),
                                      indexed_rows(tuple(tuple(row) for row in rows)), len(seed)-95)
        if self.catalogue is None:
            return domains, None
        triggers, images, rows, cap = self.catalogue
        from r35_pose_departures import propagate
        result = list(domains)
        removed = 0
        full = (1 << (3*len(result)))-1
        while True:
            signature = tuple((c,d) for c,d in enumerate(result) if (c,d) in triggers)
            if not signature:
                return (result,dict(over=False,removed=removed)) if removed else (domains,None)
            if signature != self.last_signature:
                # Use only the held facts to derive these implications. Recording
                # the other concrete domains here would be unsound across siblings.
                abstract = [full]*len(result)
                for c,bit in signature:
                    abstract[c] = bit
                narrowed,receipt = propagate(abstract,images,rows,cap)
                cuts = tuple((c,old & ~new) for c,(old,new) in enumerate(zip(abstract,narrowed)) if old != new)
                self.last_signature = signature
                self.last_actions = receipt['over'],cuts
            conflict,cuts = self.last_actions
            if conflict:
                return result,dict(over=True,removed=removed)
            before = removed
            for c,bits in cuts:
                removed += (result[c]&bits).bit_count()
                result[c] &= ~bits
                if not result[c]:
                    return result,dict(over=True,removed=removed)
            if removed == before:
                return result,dict(over=False,removed=removed)


def screen_cubes(cubes, receiver, stop=None):
    """Use only on newly owned, unstarted cubes, before a native owner is created."""
    retained=[]
    closed=removed=0
    for index,cube in enumerate(cubes):
        if stop is not None and stop():
            retained.extend(cubes[index:])
            break
        narrowed,result=receiver.prune(cube)
        if result is not None:
            removed+=result['removed']
            if result['over']:
                closed+=1
                continue
        retained.append(tuple(narrowed))
    return retained,closed,removed

def validate_receiver(algorithm, options, target, resume):
    if algorithm not in ('dfs','gpu-domains') or target!='gold' or options.get('symbolic_frontier'):
        raise ValueError('Reference pose cuts require DFS or GPU domain search for Gold.')
    if algorithm=='dfs' and (options.get('shared_coverage') or options.get('reason_tracking') or options.get('lcg') or
                             isinstance(resume,dict) and 'parallel_version' in resume):
        raise ValueError('Reference pose cuts require the single-process DFS without reason tracking or LCG.')
    if isinstance(resume,dict) and resume.get('schema')=='diamond-symbolic-coordinator-v1':
        raise ValueError('Reference pose cuts cannot be enabled on a symbolic frontier.')


def prepare_frontier(problem, frontier, stop=None):
    from constraint_work import WorkScope,frontier_snapshot,restore_frontier
    scope=WorkScope.for_problem(problem,'gold')
    cubes=restore_frontier(frontier,scope) if frontier is not None else (scope.root,)
    retained,closed,removed=screen_cubes(cubes,FrontierPoses(problem),stop)
    if closed or removed:
        original=WorkScope.restore(frontier['scope']) if frontier is not None else scope
        frontier=frontier_snapshot(original,retained)
    return frontier,dict(reference_pose_closed=closed,reference_pose_removed=removed)
