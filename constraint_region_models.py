"""Whole-board CP-SAT variables selecting exact regional alternatives.

A deterministic disjoint partition covers all cells. Each table contains only
complete concrete regional histories, not merged placement-support unions.
An over-budget multi-cell table is discarded and those cells get singleton
regions; this loses strength without excluding a possible arrangement. Existing
cross-region seam, physical-inventory and Gold constraints remain in force.
"""
from constraint_regions import HierarchicalRegionPruner
from constraint_region_network import ExactRegionDiagram
from constraint_separator import _Budget


def add_regional_alternatives(problem,domains,model,codes,*,target='gold',check=lambda:None,
                              max_cells=3,max_rows=4096,max_steps=200000):
    if type(max_cells)is not int or not 1<=max_cells<=16:raise ValueError('Invalid model region size.')
    if type(max_rows)is not int or max_rows<1 or type(max_steps)is not int or max_steps<1:
        raise ValueError('Invalid model regional table budget.')
    masks=[sum(1<<code for code in row) for row in domains]
    owner=HierarchicalRegionPruner(problem,target);pending=set(range(problem.n));index=0
    stats={'regional_model_regions':0,'regional_model_joined_regions':0,
           'regional_model_rows':0,'regional_model_diagram_nodes':0,'regional_model_budget_fallbacks':0}
    while pending:
        check();first=min(pending,key=lambda c:(len(domains[c]),c));cells=[first]
        while len(cells)<max_cells:
            candidates=set().union(*(owner.neighbors[c] for c in cells))&pending-set(cells)
            if not candidates:break
            cells.append(min(candidates,key=lambda c:(len(domains[c]),c)))
        selected=None
        while len(cells)>1:
            try:selected=ExactRegionDiagram(owner,cells,masks,check,max_rows=max_rows,max_steps=max_steps);break
            except _Budget:
                stats['regional_model_budget_fallbacks']+=1;cells.pop()
        if selected is None:
            cells=[first];rows=[(code,) for code in domains[first]]
        else:
            cells=list(selected.cells);rows=[row for row,signature in selected.histories(check)]
            stats['regional_model_diagram_nodes']+=len(selected.nodes)+len(selected.terminals)
            stats['regional_model_joined_regions']+=1
        choice=model.new_int_var(0,max(0,len(rows)-1),f'region_choice_{index}')
        # The relation exposes the regional alternative as a real model
        # variable. Whole physical-tile and seam constraints link the selected
        # cell values across regions; arbitrary mixed rows are never admitted.
        model.add_allowed_assignments([choice,*[codes[c] for c in cells]],
                                      [(i,*row) for i,row in enumerate(rows)])
        stats['regional_model_regions']+=1;stats['regional_model_rows']+=len(rows)
        pending.difference_update(cells);index+=1
    return stats
