"""Bounded optional reasoning stages shared by exact DFS slices.

Stages only publish proved consequences. Completed internal caches survive warm
slices; cancelled or capped queries remain unknown.
"""
from constraint_algebra import AffinePruner, STAT_KEYS as AFFINE_STATS
from constraint_pb import PBLearner, STAT_KEYS as PB_STATS
from constraint_interface import InterfaceProjector, STAT_KEYS as INTERFACE_STATS
from constraint_region_research import RegionalResearchPruner, STAT_KEYS as REGION_STATS

STAT_KEYS = AFFINE_STATS + PB_STATS + INTERFACE_STATS + REGION_STATS
KEYS = ('affine_reasoning','pb_learning','obstruction_rules','interface_formulas') + RegionalResearchPruner.KEYS

class ResearchPruner:
    def __init__(self,problem,target,options):
        self.options={key:bool(options[key]) for key in KEYS}
        self.stages=[]
        if options['affine_reasoning']:self.stages.append(('affine port equations',AffinePruner(problem)))
        if options['pb_learning'] or options['obstruction_rules']:
            self.stages.append(('verified integer inequalities',PBLearner(problem,derive=options['pb_learning'],discover=options['obstruction_rules'])))
        if options['interface_formulas']:self.stages.append(('verified regional interface formulas',InterfaceProjector(problem)))
        if any(options[key] for key in RegionalResearchPruner.KEYS):
            self.stages.append(('bounded regional reasoning',RegionalResearchPruner(problem,target,options)))
    @property
    def stats(self):
        result=dict.fromkeys(STAT_KEYS,0)
        for _,stage in self.stages:result.update(stage.stats)
        return result
    def prune(self,domains,check,bulk):
        for rule,stage in self.stages:
            check()
            if not all(domains):break
            bulk(lambda:stage.prune(domains,check),rule)
