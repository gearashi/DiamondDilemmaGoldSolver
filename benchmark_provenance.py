"""Frozen source and local runtime identity for reproducible benchmark reports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import platform
import subprocess

SOURCE_SUFFIXES=frozenset(('.py','.c','.cu','.wgsl','.html'))
NATIVE_SUFFIXES=frozenset(('.dll','.so','.dylib','.pyd','.json'))
DEPENDENCIES=('numpy','python-sat','ortools','cupy-cuda12x','wgpu','setuptools')


def source_snapshot(root):
    """Include fixture/test code and local compiled artifacts, not just imports.

    Root-relative names avoid machine-specific directory names. Canonical LF
    hashes apply only to text; binary identity always uses the original bytes.
    Existing artifacts must remain unchanged throughout a final comparison.
    """
    root=Path(root)
    paths={p for p in root.iterdir() if p.is_file() and
           (p.suffix in SOURCE_SUFFIXES or p.name.startswith('requirements') and p.suffix=='.txt')}
    workflow=root/'.github/workflows/tests.yml'
    if workflow.is_file():paths.add(workflow)
    native=root/'.native'
    if native.is_dir():paths.update(p for p in native.iterdir() if p.is_file() and p.suffix in NATIVE_SUFFIXES)
    result={}
    for path in sorted(paths):
        data=path.read_bytes();record={'raw_sha256':hashlib.sha256(data).hexdigest()}
        if path.suffix in SOURCE_SUFFIXES|{'.txt','.json','.yml'}:
            record['lf_sha256']=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
        result[path.relative_to(root).as_posix()]=record
    return result


def runtime_snapshot():
    """Versions plus the SAT extension identity without importing optional engines."""
    versions={}
    for name in DEPENDENCIES:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result={'python':sys.version,'executable':sys.executable,'machine':platform.machine(),'argv':list(sys.argv),'packages':versions}
    spec=importlib.util.find_spec('pysolvers')
    if spec is not None and spec.origin:
        path=Path(spec.origin)
        if path.is_file():result['sat_native']={'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    package=importlib.util.find_spec('pysat')
    if package is not None and package.submodule_search_locations:
        wrapper=Path(next(iter(package.submodule_search_locations)))/'solvers.py'
        if wrapper.is_file():result['sat_wrapper']={'name':'pysat/solvers.py','sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest()}
    return result


def repository_snapshot(root):
    """HEAD identifies history; dirty paths prevent claiming unchanged commit bytes."""
    try:
        head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,timeout=15)
        status=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=root,capture_output=True,text=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired) as error:
        return {'available':False,'reason':type(error).__name__}
    if head.returncode or status.returncode:return {'available':False,'reason':'No readable Git checkout'}
    return {'available':True,'commit':head.stdout.strip(),'dirty':bool(status.stdout.strip()),'status':status.stdout.splitlines()}
