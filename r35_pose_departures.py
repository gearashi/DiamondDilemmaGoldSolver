"""Propagate proved departures from oriented reference poses in geographic Gold."""
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType


@lru_cache(maxsize=1)
def indexed_rows(rows):
    by_cell = {}
    for cells in rows:
        mask = sum(1 << c for c in cells)
        for c in cells:
            by_cell.setdefault(c, []).append(mask)
    return MappingProxyType({c:tuple(items) for c,items in by_cell.items()})


def cell_indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask -= bit


def propagate(domains, images, rows, max_kept):
    """Each row forbids keeping all its poses. An image maps reference cells to query cells."""
    narrowed = list(domains)
    result = dict(over=False,removed=0,passes=0,clause_checks=0,distance_units=0)
    if not all(narrowed):
        result.update(over=True,reason='empty_domain')
        return narrowed,result
    while True:
        before = result['removed']
        result['passes'] += 1
        for rotation_id,image in enumerate(images):
            fixed = possible = 0
            for c,(cell,pose) in enumerate(image):
                if narrowed[cell] & pose:
                    possible |= 1 << c
                    if narrowed[cell] == pose:
                        fixed |= 1 << c
            if not fixed and max_kept > 0:
                continue
            nfixed = fixed.bit_count()
            if nfixed > max_kept:
                result.update(over=True,reason='cell_distance',forced=nfixed,rotation=rotation_id)
                return narrowed,result
            remove = possible ^ fixed if nfixed == max_kept else 0
            result['distance_units'] += remove.bit_count()
            inspected = set()
            for c in cell_indices(fixed):
                for row in rows.get(c,()):
                    if row in inspected:
                        continue
                    inspected.add(row)
                    if row & ~possible:
                        continue
                    result['clause_checks'] += 1
                    missing = row & ~fixed
                    if not missing:
                        result.update(over=True,reason='pose_clause',rotation=rotation_id,
                                      reference_row=list(cell_indices(row)),
                                      fixed_poses=[(image[i][0],image[i][1].bit_length()-1) for i in cell_indices(row)])
                        return narrowed,result
                    if not (missing & (missing-1)):
                        remove |= missing
            for c in cell_indices(remove):
                cell,pose = image[c]
                narrowed[cell] &= ~pose
                result['removed'] += 1
                if not narrowed[cell]:
                    result.update(over=True,reason='empty_domain')
                    return narrowed,result
        if result['removed'] == before:
            return narrowed,result


@lru_cache(maxsize=1)
def reference_images(seed):
    from constraint_symmetry import proper_rotations
    from geometry import build_board
    images = []
    for rotation in proper_rotations(build_board()):
        codes = rotation.apply(seed)
        images.append(tuple((cell,1 << codes[cell]) for cell in rotation.cells))
    return tuple(images)


def proof_version(directory):
    versions = []
    for name in ('R35_REPAIR_PLACEMENTS.json','R35_REPAIR_REFINED.json','R35_REPAIR_REGIONAL.json'):
        stat = (directory/name).stat()
        versions.append((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
    return tuple(versions)


def checked_profile(problem):
    from r35_repair_seeds import active_catalog, known_hint_conflicts
    directory = Path(__file__).resolve().parent/'docs'
    try:
        version = proof_version(directory)
    except OSError:
        return None
    guards = tuple(map(tuple,known_hint_conflicts()))
    key = (directory,version,guards)
    cached = getattr(problem,'_geographic_pose_profile',None)
    if cached is not None and cached[0] == key:
        return cached[1]
    # File identity only reuses a previously authenticated theorem on a freshly
    # verified mesh. It never authorizes new proof rows.
    record = active_catalog(problem)
    if record is None or record.get('lower_bound') != 95:
        return None
    # Four later independent conflicts exclude complete master hits. The
    # distance certificate itself uses only the original 4,225 rows.
    rows = tuple(tuple(cells) for cells in (*record['rows'],*record.get('post_master',())))
    if not rows or any(len(row)<2 for row in rows):
        return None
    profile = (reference_images(tuple(record['codes'])),indexed_rows(rows),problem.n-record['lower_bound'])
    try:
        if proof_version(directory) == version:
            problem._geographic_pose_profile = (key,profile)
    except OSError:
        pass
    return profile


def dense_tile_support(domains):
    """Certify all remaining cell/tile edges by greedy extension of any chosen edge."""
    starts = ((1 << (3*len(domains)))-1)//7
    fixed = 0
    free = []
    for mask in domains:
        tiles = (mask | mask >> 1 | mask >> 2) & starts
        if not tiles:
            return False
        if tiles & (tiles-1):
            free.append(tiles)
        elif tiles & fixed:
            return False
        else:
            fixed |= tiles
    if any(row & fixed for row in free):
        return False
    # Fix any one free cell and tile. Each other row can lose at most
    # one option. In increasing row-size order the i-th remaining row
    # still has at least i options (one-based), so greed always succeeds.
    sizes = sorted(row.bit_count() for row in free)
    return all(size >= i+2 for i,size in enumerate(sizes[:-1]))


def crossover_seams(problem,domains,images):
    from r35_pose_matching import tight_support
    width = 3*problem.n
    full = (1 << width)-1
    if all(d == full for d in domains):
        return domains,{'removed':0,'over':False}
    engine = getattr(problem,'_original_crossover_seams',None)
    if engine is None:
        from r32_gold_bonds import inventory
        from r35_repair_propagation import RepairPropagation
        partners = [[] for _ in range(3*problem.n)]
        for a,b in inventory(problem).possible:
            partners[a].append(b)
            partners[b].append(a)
        engine = RepairPropagation(problem,partners)
        problem._original_crossover_seams = engine
    # Matching identifies cell/tile pairs that occur in no assignment of the
    # physical tiles. A deletion can expose an unsupported seam and vice versa.
    seed = tuple(pose.bit_length()-1 for _,pose in sorted(images[0]))
    current = domains
    matchings = 0
    while True:
        narrowed = engine.narrow((),(),width,True,domains=current)
        info = dict(removed=0,over=False,domain_width=width,tile_matchings=matchings)
        if narrowed is None:
            info.update(over=True,reason='image_pool_seams')
            return domains,info
        if dense_tile_support(narrowed):
            info['dense_tile_support'] = True
            info['removed'] = sum((a ^ b).bit_count() for a,b in zip(domains,narrowed))
            return narrowed,info
        supported = tight_support(narrowed,seed)
        matchings += 1
        info['tile_matchings'] = matchings
        if supported is None:
            info.update(over=True,reason='geographic_tile_matching')
            return domains,info
        if all(a==b for a,b in zip(narrowed,supported)):
            info['removed'] = sum((a ^ b).bit_count() for a,b in zip(domains,narrowed))
            return narrowed,info
        current = supported


def original_pose_departures(problem, domains):
    # An open root gives no reference literal or geographic restriction.
    if problem.n != 160:
        return domains,None
    full = (1 << (3*problem.n))-1
    if all(mask == full for mask in domains):
        return domains,None
    from r35_claim_catalogue import published_scope
    if not published_scope(problem):
        return domains,None
    profile = checked_profile(problem)
    if profile is None:
        return domains,None
    from r35_pose_matching import owner_bounds
    from r35_pose_image_pairs import pair_bounds
    images,indexed,limit = profile
    current = domains
    stats = dict(over=False, removed=0, passes=0, clause_checks=0, distance_units=0)
    steps = []
    filters = (
        ('two_reference_images',partial(pair_bounds,images=images,cap=limit)),
        ('crossover_seams',partial(crossover_seams,problem,images=images)),
        ('tile_assignment',partial(owner_bounds,images=images,max_kept=limit)),
    )
    last_unchanged = {}
    while True:
        current,units = propagate(current,images,indexed,limit)
        for key in ('passes','removed','clause_checks','distance_units'):
            stats[key] += units[key]
        if units['over']:
            stats.update((k,v) for k,v in units.items() if k not in ('passes','removed','clause_checks','distance_units'))
            break
        before = stats['removed']
        for name, apply in filters:
            snapshot = tuple(current)
            if last_unchanged.get(name) == snapshot:
                continue
            current,info = apply(current)
            if name == 'crossover_seams' and not info['over']:
                last_unchanged[name] = tuple(current)  # already at seam/matching closure
            elif not info['removed'] and not info['over']:
                last_unchanged[name] = snapshot
            if any(info.values()):
                steps.append(dict(stage=name,**info))
            stats['removed'] += info['removed']
            if info['over'] or not all(current):
                stats.update(over=True,reason=info.get('reason','empty_domain'))
                break
        if stats['over'] or stats['removed']==before:
            break
    if steps:
        stats['support_steps'] = steps
    return current,stats
