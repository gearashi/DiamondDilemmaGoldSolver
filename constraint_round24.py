"""Bounded optional reasoning stages shared by exact DFS slices.

Stages only publish proved consequences. Completed internal caches survive warm
slices; cancelled or capped queries remain unknown.
"""
from constraint_algebra import AffinePruner, STAT_KEYS as AFFINE_STATS
from constraint_affine_reasons_round26 import STAT_KEYS as AFFINE_REASON_STATS
from constraint_pb import PBLearner, STAT_KEYS as PB_STATS
from constraint_arithmetic_round26 import STAT_KEYS as ARITHMETIC_STATS
from constraint_conflict_pb import ConflictPBLearner, STAT_KEYS as CONFLICT_PB_STATS
from constraint_lattice import IntegerLatticePruner, STAT_KEYS as LATTICE_STATS
from constraint_lattice_round26 import ContributionLatticePruner, TREE_KEYS as LATTICE_TREE_STATS
from constraint_interface import InterfaceProjector, STAT_KEYS as INTERFACE_STATS
from constraint_region_research import RegionalResearchPruner, STAT_KEYS as REGION_STATS

STAT_KEYS = AFFINE_STATS + PB_STATS + INTERFACE_STATS + REGION_STATS + CONFLICT_PB_STATS + LATTICE_STATS + LATTICE_TREE_STATS + ARITHMETIC_STATS + AFFINE_REASON_STATS
MAX_GAUGE_KEYS = frozenset(('factor_live_piece_width','compact_factor_frontier_width','compact_factor_live_resources','compact_gold_port_peak'))
KEYS = ('smart_learning','pb_row_queue','lattice_tree','integer_lattice','conflict_pb','incremental_affine','affine_reasoning','pb_learning','obstruction_rules','interface_formulas') + RegionalResearchPruner.KEYS

class ResearchPruner:
    def __init__(self,problem,target,options):
        self.options={key:bool(options[key]) for key in KEYS}
        self.stages=[]
        if options['affine_reasoning']:self.stages.append(('affine port equations',AffinePruner(problem,incremental=options['incremental_affine'],explain=options['smart_learning'])))
        if options['pb_learning'] or options['obstruction_rules']:
            self.stages.append(('verified integer inequalities',PBLearner(problem,derive=options['pb_learning'],discover=options['obstruction_rules'],indexed=options['pb_row_queue'],explain=options['smart_learning'])))
        if options['integer_lattice']:self.stages.append(('exact integer seam and inventory lattice',(ContributionLatticePruner if options['lattice_tree'] else IntegerLatticePruner)(problem)))
        if options['conflict_pb']:self.stages.append(('conflict-directed integer cuts',ConflictPBLearner(problem,explain=options['smart_learning'])))
        if options['interface_formulas']:self.stages.append(('verified regional interface formulas',InterfaceProjector(problem)))
        if any(options[key] for key in RegionalResearchPruner.KEYS):
            self.stages.append(('bounded regional reasoning',RegionalResearchPruner(problem,target,options)))
    def configure_cache_dir(self,path):
        for _,stage in self.stages:
            configure=getattr(stage,'configure_cache_dir',None)
            if configure is not None:configure(path)

    @property
    def solution(self):
        for _,stage in self.stages:
            candidate=getattr(stage,'solution',None)
            if candidate is not None:return candidate
        return None

    @property
    def stats(self):
        result=dict.fromkeys(STAT_KEYS,0)
        for _,stage in self.stages:
            result.update(stage.stats)
            if hasattr(stage,'queue_stats'):result.update(stage.queue_stats)
            if hasattr(stage,'reason_stats'):result.update(stage.reason_stats)
        return result
    def drain_deduction_chains(self,target='gold',limit=8):
        if target not in ('gold','edges') or type(limit)is not int or not 0<=limit<=128:raise ValueError('Invalid reasoning proof export policy.')
        emitted=0
        for _,stage in self.stages:
            drain=getattr(stage,'drain_deduction_chains',None)
            if drain is None or emitted>=limit:continue
            iterator=iter(drain(target=target,limit=limit-emitted))
            try:
                for certificate in iterator:
                    yield certificate
                    emitted+=1
            finally:
                close=getattr(iterator,'close',None)
                if close is not None:close()

    def prune(self,domains,check,bulk):
        for rule,stage in self.stages:
            check()
            if not all(domains):break
            bulk(lambda:stage.prune(domains,check),rule)
