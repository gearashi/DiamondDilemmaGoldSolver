"""Public algorithm identifiers; the default is a practical choice, not a speed claim."""
ALGORITHMS = ('gpu-dfs', 'dfs', 'cp', 'cp-sat', 'sat', 'hybrid')
RECOMMENDED = 'dfs'  # Strong pruning + exact resume; bounded comparison has no proven fastest engine.
LABELS = {'gpu-dfs':'GPU exhaustive DFS (saved branches)', 'dfs':'DFS with constraint pruning', 'cp':'Constraint programming', 'cp-sat':'CP-SAT', 'sat':'SAT (Glucose)', 'hybrid':'Experimental GPU sampling + CPU DFS'}
