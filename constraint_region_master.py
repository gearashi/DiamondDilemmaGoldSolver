"""On-demand regional search and independently checked LP infeasibility.

Optional OR-Tools GLOP supplies candidate dual vectors. Exact Fraction checks,
not solver status or floating tolerances, authorize every negative conclusion.
The pricing master is an inventory relaxation: feasible means only possible.
"""
from fractions import Fraction
from collections import OrderedDict
from constraint_separator import _Budget
from constraint_pruning import values
from validator import reverse_mask


def check_farkas(columns,rhs,multipliers):
    """A x=b, x>=0: y.b>0 and y.A<=0 certify that no x exists."""
    if len(rhs)!=len(multipliers):raise ValueError('Certificate dimension mismatch.')
    y=tuple(Fraction(x) for x in multipliers);b=tuple(Fraction(x) for x in rhs)
    if sum(a*c for a,c in zip(y,b))<=0:return False
    return all(sum(y[i]*Fraction(value) for i,value in column.items())<=0 for column in columns)


def phase_one(columns,rhs,check=lambda:None,time_ms=1000,telemetry=None):
    """Return an untrusted candidate dual; callers verify it exactly."""
    try:from ortools.linear_solver import pywraplp
    except ImportError:return {'status':'unavailable'}
    check();solver=pywraplp.Solver.CreateSolver('GLOP')
    if solver is None:return {'status':'unavailable'}
    if telemetry is not None:telemetry['builds']=telemetry.get('builds',0)+1
    solver.SetTimeLimit(int(time_ms));constraints=[solver.Constraint(float(b),float(b)) for b in rhs]
    variables=[]
    for j,column in enumerate(columns):
        check();x=solver.NumVar(0,solver.infinity(),f'x{j}');variables.append(x)
        for i,value in column.items():constraints[i].SetCoefficient(x,float(value))
    objective=solver.Objective()
    for i,constraint in enumerate(constraints):
        plus=solver.NumVar(0,solver.infinity(),f'p{i}');minus=solver.NumVar(0,solver.infinity(),f'm{i}')
        constraint.SetCoefficient(plus,1);constraint.SetCoefficient(minus,-1)
        objective.SetCoefficient(plus,1);objective.SetCoefficient(minus,1)
    objective.SetMinimization()
    if telemetry is not None:telemetry['calls']=telemetry.get('calls',0)+1
    status=solver.Solve()
    if status!=pywraplp.Solver.OPTIMAL:result={'status':'unknown'}
    else:
        dual=tuple(Fraction(c.dual_value()).limit_denominator(1000000) for c in constraints)
        result={'status':'optimal','objective':objective.Value(),'dual':dual,
                'primal':tuple(x.solution_value() for x in variables)}
    # A controller may commit this completed query even when the following
    # cooperative Stop prevents normal return. Never persist native objects.
    if telemetry is not None:telemetry['result']=result
    check();return result


def puzzle_linear_system(problem,domains,cells=None,check=lambda:None,max_columns=1024):
    """Joint placement/inventory/seam-pattern equality relaxation."""
    cells=tuple(range(problem.n)) if cells is None else tuple(sorted(set(cells)))
    choices={cell:tuple(values(domains[cell])) for cell in cells};rows=[];rhs=[]
    def row(coefficients,value):rows.append(coefficients);rhs.append(value)
    for cell in cells:row({(cell,code):1 for code in choices[cell]},1)
    pieces=sorted({code//3 for c in cells for code in choices[c]})
    for tile in pieces:
        coefficients={(cell,code):1 for cell in cells for code in choices[cell] if code//3==tile}
        # Slack converts <=1 to an equality for a selected proper subset.
        if len(cells)!=problem.n:coefficients[('unused',tile)]=1
        row(coefficients,1)
    for a,sa,b,sb in problem.edges:
        if a not in choices or b not in choices:continue
        patterns={int(problem.masks[x,sa]) for x in choices[a]}|{reverse_mask(int(problem.masks[x,sb])) for x in choices[b]}
        for pattern in patterns:
            coefficients={(a,x):1 for x in choices[a] if int(problem.masks[x,sa])==pattern}
            coefficients.update({(b,x):-1 for x in choices[b] if reverse_mask(int(problem.masks[x,sb]))==pattern})
            row(coefficients,0)
    keys=tuple(dict.fromkeys(key for coefficients in rows for key in coefficients))
    if len(keys)>max_columns:raise _Budget
    columns=[]
    for key in keys:
        check();columns.append({i:coefficients[key] for i,coefficients in enumerate(rows) if key in coefficients})
    return columns,tuple(rhs),keys


class CertifiedRelaxationPruner:
    def __init__(self,problem,max_cells=8,max_domain=8,max_columns=1024):
        self.problem=problem;self.max_cells=max_cells;self.max_domain=max_domain;self.max_columns=max_columns
        self.stats={'certified_lp_queries':0,'certified_lp_failures':0,'certified_lp_unknown':0,
                    'certified_lp_unverified_duals':0};self.last_certificate=None
    def prune(self,domains,check=lambda:None,full=False):
        cells=tuple(range(self.problem.n)) if full else tuple(sorted(
            (c for c,d in enumerate(domains) if 1<d.bit_count()<=self.max_domain),key=lambda c:(domains[c].bit_count(),c))[:self.max_cells])
        if len(cells)<2:return False
        try:columns,rhs,keys=puzzle_linear_system(self.problem,domains,cells,check,self.max_columns)
        except _Budget:self.stats['certified_lp_unknown']+=1;return False
        self.stats['certified_lp_queries']+=1;result=phase_one(columns,rhs,check)
        if result['status']!='optimal':self.stats['certified_lp_unknown']+=1;return False
        if check_farkas(columns,rhs,result['dual']):
            self.last_certificate={'cells':cells,'domains':tuple(domains[c] for c in cells),
                'multipliers':tuple((x.numerator,x.denominator) for x in result['dual'])}
            self.stats['certified_lp_failures']+=1;domains[cells[0]]=0;return True
        if result['objective']>0:self.stats['certified_lp_unverified_duals']+=1
        return False


def diagram_max_weight(diagram,piece_weights,check=lambda:None):
    """Exact dynamic-programming pricing; no assignment-row enumeration."""
    memo={};steps=0
    def visit(node):
        nonlocal steps
        check()
        if node in memo:return memo[node]
        if node in diagram.terminals:return (Fraction(0),())
        choices=[]
        for code,child in diagram.nodes[node][1]:
            steps+=1;score,path=visit(child)
            choices.append((score+Fraction(piece_weights.get(code//3,0)),(code,*path)))
        result=max(choices,key=lambda item:(item[0],item[1]));memo[node]=result;return result
    if diagram.root is None:return None,(),steps
    score,path=visit(diagram.root);return score,path,steps


def priced_inventory_master(diagrams,n_pieces,check=lambda:None,max_rounds=100,time_ms=1000):
    """Phase-I column generation with completed exact diagram pricing.

    One column chooses one regional history's complete inventory. All regions
    must cover the whole board and consume n_pieces physical pieces. Cross-region
    seam/Gold conditions are omitted, so infeasibility is sound and feasibility
    is only a relaxation result. Empty restricted masters never prove failure.
    """
    region_count=len(diagrams);rhs=(1,)*(region_count+n_pieces);columns=[];seen=set();pricing_steps=0
    if sum(len(d.cells) for d in diagrams)!=n_pieces or len(set(c for d in diagrams for c in d.cells))!=n_pieces:
        raise ValueError('Pricing regions must form a complete disjoint board partition.')
    for iteration in range(max_rounds):
        check();result=phase_one(columns,rhs,check,time_ms)
        if result['status']!='optimal':return {'status':'unknown','rounds':iteration,'columns':len(columns),'pricing_steps':pricing_steps}
        y=result['dual'];added=False;priced=[]
        for region,diagram in enumerate(diagrams):
            score,path,steps=diagram_max_weight(diagram,{t:y[region_count+t] for t in range(n_pieces)},check)
            pricing_steps+=steps
            if score is None:priced.append(None);continue
            value=y[region]+score;priced.append(value)
            if value>0:
                inventory=sum(1<<(code//3) for code in path);key=(region,inventory)
                if key not in seen:
                    seen.add(key);columns.append({region:1,**{region_count+t:1 for t in values(inventory)}});added=True
        if added:continue
        if sum(y)>0 and all(value is None or value<=0 for value in priced) and check_farkas(columns,rhs,y):
            return {'status':'infeasible','rounds':iteration+1,'columns':len(columns),'pricing_steps':pricing_steps,
                    'certificate':tuple((x.numerator,x.denominator) for x in y)}
        if result['objective']<=0:
            return {'status':'possible','rounds':iteration+1,'columns':len(columns),'pricing_steps':pricing_steps}
        return {'status':'unknown','rounds':iteration+1,'columns':len(columns),'pricing_steps':pricing_steps}
    return {'status':'unknown','rounds':max_rounds,'columns':len(columns),'pricing_steps':pricing_steps}


class LazyRegionMaster:
    """Generate local realizations only when demanded by a regional master.

    The master state is the exact used inventory and strand/seam interface.
    Completed failed interfaces are Benders-style exclusions shared by all
    internal histories with that interface. It solves existence, not all-value
    support in one query. A cap returns unknown and publishes no failed parent.
    """
    def __init__(self,owner,domains,regions,max_steps=50000):
        self.owner=owner;self.domains=tuple(domains);self.regions=tuple(tuple(sorted(r)) for r in regions)
        self.max_steps=max_steps;self.stats={'lazy_steps':0,'lazy_local_rows':0,'lazy_interface_hits':0,'lazy_interface_cuts':0}
        flattened=tuple(c for r in self.regions for c in r)
        if not flattened or len(set(flattened))!=len(flattened):raise ValueError('Regions must be nonempty and disjoint.')
        self.failed=set();self.solution=None
    def solve(self,check=lambda:None):
        from constraint_region_network import assignment_signature,joined_signature
        def tick():
            self.stats['lazy_steps']+=1
            if self.stats['lazy_steps']>self.max_steps:raise _Budget
            check()
        def local_rows(cells,used,previous):
            chosen={}
            def walk(i,taken):
                tick()
                if i==len(cells):
                    codes=tuple(chosen[c] for c in cells)
                    signature=assignment_signature(self.owner,cells,codes,self.domains,check)
                    if signature is not None:self.stats['lazy_local_rows']+=1;yield codes,signature
                    return
                cell=cells[i]
                for code in values(self.domains[cell]):
                    if taken>>(code//3)&1:continue
                    all_chosen={**previous,**chosen}
                    if any(other in all_chosen and self.owner.masks[code][side]!=reverse_mask(self.owner.masks[all_chosen[other]][oside]) for other,side,oside,_ in self.owner.adj[cell]):continue
                    chosen[cell]=code;yield from walk(i+1,taken|(1<<(code//3)));del chosen[cell]
            yield from walk(0,used)
        def search(index,signature,assignment):
            tick();key=(index,signature)
            if key in self.failed:self.stats['lazy_interface_hits']+=1;return False
            if index==len(self.regions):self.solution=dict(assignment);return True
            cells=self.regions[index];previous=tuple(c for r in self.regions[:index] for c in r)
            for codes,local in local_rows(cells,0 if signature is None else signature[0],assignment):
                merged=local if signature is None else joined_signature(self.owner,previous,signature,cells,local,self.domains,check)
                if merged is not None and search(index+1,merged,{**assignment,**dict(zip(cells,codes))}):return True
            self.failed.add(key);self.stats['lazy_interface_cuts']+=1;return False
        try:return search(0,None,{})
        except _Budget:return None
