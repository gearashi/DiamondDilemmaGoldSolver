"""Opt-in integration of bounded R24 regional reasoning experiments.

Every stage either publishes completed consequences, reports unknown, or records
an explicit admission skip. The existence-only rank and LP relaxations never
supply unsupported placement masks. All selected-cell queries are optimistic
outside their stated scope; the Gold routing stage runs only for Gold targets.
"""
from constraint_regions import HierarchicalRegionPruner
from constraint_region_elimination import EliminationPruner
from constraint_region_symbolic import inventory_supports,subset_inventory_supports,representative_component_feasible
from constraint_region_master import LazyRegionMaster,CertifiedRelaxationPruner
from constraint_region_oracles import LabelledRoutingPruner
from constraint_separator import _Budget
from constraint_region_incremental import IncrementalRegionCache, STAT_KEYS as RESTRICT_STATS
from constraint_region_compact import CompactFactorPruner, STAT_KEYS as COMPACT_STATS
from constraint_region_andor import SeparatorSearch, STAT_KEYS as ANDOR_STATS
from constraint_region_refinement import SelectiveRefinementPruner, STAT_KEYS as SELECTIVE_STATS
from constraint_region_joint import JointSymbolicPruner, STAT_KEYS as JOINT_STATS
from constraint_region_rank import AlgebraicConnectivity, STAT_KEYS as RANK_STATS
from constraint_region_isomorphism import IsomorphicResidualPruner, STAT_KEYS as ISO_STATS
from constraint_region_pricing import GoldPricingPruner, STAT_KEYS as PRICING_STATS
from constraint_region_multichain import MultichainRoutingPruner, STAT_KEYS as MULTICHAIN_STATS
from constraint_region_certificate import CertifiedRegionCache, STAT_KEYS as CERTIFICATE_STATS


STAT_KEYS=('regional_research_queries','regional_research_completed','regional_research_unknown',
    'regional_research_admission_skips','regional_research_prunes','regional_research_lazy_rows',
    'regional_research_lazy_cuts','regional_research_subset_operations','regional_research_rank_rows_removed',
    'regional_research_rank_steps','factor_queries','factor_steps','factor_prunes','factor_cache_hits',
    'factor_unknown','factor_dd_merges','factor_live_piece_width','factor_forgotten_resources',
    'certified_lp_queries','certified_lp_failures','certified_lp_unknown','certified_lp_unverified_duals',
    'labelled_queries','labelled_steps','labelled_unknown','labelled_failures','labelled_admission_skips') + RESTRICT_STATS + COMPACT_STATS + ANDOR_STATS + SELECTIVE_STATS + JOINT_STATS + RANK_STATS + ISO_STATS + PRICING_STATS + MULTICHAIN_STATS + CERTIFICATE_STATS


class RegionalResearchPruner:
    KEYS=('region_certificates','multichain_routing','gold_pricing','residual_isomorphism','algebraic_connectivity','joint_symbolic_regions','selective_refinement','separator_search','gold_elimination','compact_factors','durable_regions','incremental_regions','elimination','mini_buckets','relaxed_diagrams','representative_connectivity',
          'subset_joins','lazy_regions','certified_relaxation','labelled_routing')
    def __init__(self,problem,target='gold',options=None,max_steps=50000):
        self.problem=problem;self.target=target;self.options=dict(options or {});self.max_steps=max_steps
        self.owner=HierarchicalRegionPruner(problem,target).enable_advanced()
        self.factors=EliminationPruner(problem,max_steps=max_steps)
        self.linear=CertifiedRelaxationPruner(problem)
        self.routing=LabelledRoutingPruner(problem,max_steps=max_steps)
        self.incremental=IncrementalRegionCache(self.owner,durable=bool(self.options.get('durable_regions')))
        self.compact=CompactFactorPruner(self.owner,gold=target=='gold' and bool(self.options.get('gold_elimination')))
        self.andor=SeparatorSearch(self.owner)
        self.selective=SelectiveRefinementPruner(self.owner)
        self.joint=JointSymbolicPruner(self.owner)
        self.algebraic=AlgebraicConnectivity(self.owner)
        self.isomorphic=IsomorphicResidualPruner(self.owner)
        self.pricing=GoldPricingPruner(self.owner)
        self.multichain=MultichainRoutingPruner(self.owner)
        self.certificates=CertifiedRegionCache(self.owner)
        self.stats=dict.fromkeys(STAT_KEYS,0)

    @property
    def solution(self):return self.compact.solution if self.compact.solution is not None else self.andor.solution

    def configure_cache_dir(self,directory):
        self.incremental.configure_cache_dir(directory)
        self.certificates.configure_cache_dir(directory)

    def _cells(self,domains):
        eligible={c for c,d in enumerate(domains) if 0<d.bit_count()<=8}
        seeds=sorted((c for c in eligible if domains[c].bit_count()>1),key=lambda c:(domains[c].bit_count(),c))
        if not seeds:return ()
        chosen=[];pending=[seeds[0]];seen=set(pending)
        while pending and len(chosen)<8:
            cell=pending.pop(0);chosen.append(cell)
            for other in sorted(self.owner.neighbors[cell]&eligible):
                if other not in seen:pending.append(other);seen.add(other)
        for cell in seeds:
            if len(chosen)>=8:break
            if cell not in seen:chosen.append(cell);seen.add(cell)
        return tuple(sorted(chosen))
    def _table_query(self,domains,cells,check,large_first=False):
        regions=((cells[:-2],cells[-2:]) if large_first and len(cells)>=6 else tuple(cells[i:i+2] for i in range(0,len(cells),2)));tables=[]
        for region in regions:
            _,result=self.owner._completed_table(domains,region,check)
            if result is None:return regions,None
            tables.append(result['table'])
        return regions,tables
    def prune(self,domains,check=lambda:None):
        if not all(domains):return False
        original=tuple(domains);changed=False
        if self.options.get('compact_factors'):
            try:changed|=self.compact.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.compact.stats)
            if self.compact.solution is not None:
                self.stats['regional_research_prunes']+=sum((a&~b).bit_count() for a,b in zip(original,domains))
                return changed
        if self.options.get('separator_search') and all(domains):
            try:changed|=self.andor.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.andor.stats)
            if self.andor.solution is not None:
                self.stats['regional_research_prunes']+=sum((a&~b).bit_count() for a,b in zip(original,domains))
                return changed
        if self.options.get('multichain_routing') and all(domains):
            try:changed|=self.multichain.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.multichain.stats)
        if self.options.get('gold_pricing') and all(domains):
            try:changed|=self.pricing.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.pricing.stats)
        if self.options.get('residual_isomorphism') and all(domains):
            try:changed|=self.isomorphic.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.isomorphic.stats)
        if self.options.get('joint_symbolic_regions') and all(domains):
            try:changed|=self.joint.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.joint.stats)
        if self.options.get('selective_refinement') and all(domains):
            try:changed|=self.selective.prune(domains,check,max_steps=self.max_steps)
            finally:self.stats.update(self.selective.stats)
        full=self.problem.n<=20 or sum(d.bit_count()>1 for d in domains)<=8
        for flag,mode in (('elimination','exact'),('mini_buckets','mini'),('relaxed_diagrams','diagram')):
            if not all(domains):break
            if self.options.get(flag):
                changed|=self.factors.prune(domains,check,mode=mode,width=4,full=full)
                if not all(domains):break
        cells=self._cells(domains) if all(domains) else ()
        if self.options.get('region_certificates') and all(domains):
            if cells:
                try:changed|=self.certificates.prune(cells,domains,check,max_steps=self.max_steps)
                finally:self.stats.update(self.certificates.stats)
            else:self.stats['regional_research_admission_skips']+=1
        if self.options.get('incremental_regions') and all(domains):
            if cells:
                try:changed|=self.incremental.prune(cells,domains,check,max_steps=self.max_steps)
                finally:self.stats.update(self.incremental.stats)
            else:self.stats['regional_research_admission_skips']+=1
        if self.options.get('certified_relaxation') and all(domains):
            changed|=self.linear.prune(domains,check,full=full)
        if self.options.get('labelled_routing') and self.target=='gold' and all(domains):
            changed|=self.routing.prune(domains,check)
        if cells and all(domains):
            if self.options.get('lazy_regions'):
                self.stats['regional_research_queries']+=1
                regions=tuple(cells[i:i+2] for i in range(0,len(cells),2))
                master=LazyRegionMaster(self.owner,domains,regions,self.max_steps)
                result=master.solve(check)
                self.stats['regional_research_lazy_rows']+=master.stats['lazy_local_rows']
                self.stats['regional_research_lazy_cuts']+=master.stats['lazy_interface_cuts']
                if result is None:self.stats['regional_research_unknown']+=1
                else:
                    self.stats['regional_research_completed']+=1
                    if not result:domains[cells[0]]=0;changed=True
            if all(domains) and (self.options.get('subset_joins') or self.options.get('representative_connectivity') or self.options.get('algebraic_connectivity')):
                self.stats['regional_research_queries']+=1
                regions,tables=self._table_query(domains,cells,check)
                if tables is None:self.stats['regional_research_unknown']+=1
                else:
                    if self.options.get('subset_joins'):
                        try:
                            supported,work=subset_inventory_supports([set(s[0] for s in table) for table in tables],check)
                            self.stats['regional_research_completed']+=1
                            self.stats['regional_research_subset_operations']+=work['operations']
                            for region,table,allowed in zip(regions,tables,supported):
                                masks=[0]*len(region)
                                for signature,(row,_) in table.items():
                                    if signature[0] in allowed:masks=[a|b for a,b in zip(masks,row)]
                                for cell,mask in zip(region,masks):domains[cell]&=mask
                        except _Budget:self.stats['regional_research_unknown']+=1
                    if (self.options.get('representative_connectivity') or self.options.get('algebraic_connectivity')) and self.target=='gold' and all(domains):
                        if len(cells)>=6:
                            regions,tables=self._table_query(domains,cells,check,large_first=True)
                        if tables is None:
                            self.stats['regional_research_unknown']+=1
                            continue_rank=False
                        else:continue_rank=True
                        if continue_rank:
                            if self.options.get('algebraic_connectivity'):
                                try:result=self.algebraic.feasible(domains,regions,tables,check,self.max_steps)
                                finally:self.stats.update(self.algebraic.stats)
                            else:
                                result,work=representative_component_feasible(self.owner,domains,regions,tables,check,self.max_steps)
                                self.stats['regional_research_rank_rows_removed']+=work['representative_rows']-work['representative_kept']
                                self.stats['regional_research_rank_steps']+=work['representative_steps']+work['representative_join_steps']
                            if result is None:self.stats['regional_research_unknown']+=1
                            else:
                                self.stats['regional_research_completed']+=1
                                if not result:domains[cells[0]]=0
        elif any(self.options.get(k) for k in ('lazy_regions','subset_joins','representative_connectivity','algebraic_connectivity')):
            self.stats['regional_research_admission_skips']+=1
        for module in (self.factors,self.linear,self.routing):self.stats.update(module.stats)
        removed=sum((a^b).bit_count() for a,b in zip(original,domains));self.stats['regional_research_prunes']+=removed
        return changed or bool(removed)
