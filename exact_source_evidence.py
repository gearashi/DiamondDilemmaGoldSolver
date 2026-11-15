"""Source closure for the two exact search runners."""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import re

from source_image_storage import source_image_writer


@lru_cache(maxsize=512)
def _source_references(name, data):
    """Parse optional function-local imports too. Exact bytes key the cache."""
    modules, resources = set(), set()
    for node in ast.walk(ast.parse(data, filename=name)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split('.', 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split('.', 1)[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*\.(?:c|cu|wgsl)', node.value):
                resources.add(node.value)
    return modules, resources


def snapshot_search_sources(root, evidence, explicit, *, include_families=True):
    """Copy the transitive local source inputs supporting an exact search result."""
    files = {path.name for path in root.iterdir() if path.is_file()}
    prefixes = ('constraint_', 'gpu_', 'domain_', 'sampling_', 'dfs_', 'r27_')
    pending = set(explicit)
    if include_families:
        pending.update(name for name in files if name.startswith(prefixes)
                       and Path(name).suffix in ('.py', '.c', '.cu', '.wgsl'))
    hashes = {}
    with source_image_writer(evidence) as write_image:
        while pending:
            name = pending.pop()
            if name in hashes:
                continue
            data = (root/name).read_bytes()
            if name.endswith('.py'):
                modules, resources = _source_references(name, data)
                pending.update(resources)
                for module in modules:
                    # Fail closed for a missing project engine imported on an optional path.
                    if module + '.py' in files or module.startswith(prefixes) or re.match(r'^r[0-9]+_', module):
                        pending.add(module + '.py')
            hashes[name] = hashlib.sha256(data).hexdigest()
            write_image(root/name,data)
    return hashes
