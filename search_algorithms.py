"""Public algorithm identifiers; the default is a practical choice, not a speed claim."""
ALGORITHMS = ('gpu-dfs', 'dfs', 'cp', 'cp-sat', 'cp-sat-circuit', 'sat', 'hybrid', 'parallel-dfs', 'sat-propagator', 'sat-channel', 'cp-sat-regions', 'gpu-domains')
RECOMMENDED = 'dfs'  # Strong pruning + exact resume; bounded comparison has no proven fastest engine.
LABELS = {'parallel-dfs':'Parallel DFS with constraint pruning', 'gpu-dfs':'GPU exhaustive DFS (saved branches)', 'dfs':'DFS with constraint pruning', 'cp':'Constraint programming', 'cp-sat':'CP-SAT', 'cp-sat-circuit':'Experimental CP-SAT direct Gold circuit', 'sat':'SAT (Glucose)', 'hybrid':'Experimental GPU sampling + repair + DFS'}

LABELS.update({'sat-propagator':'Experimental SAT with partial Gold propagation','sat-channel':'Experimental SAT tile/orientation channeling','cp-sat-regions':'Experimental CP-SAT regional alternatives'})

LABELS['gpu-domains']='Experimental CUDA domain search (128 lanes)'
