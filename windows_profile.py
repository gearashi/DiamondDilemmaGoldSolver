"""Exercise the CPU solver wheels and WebGPU library installed by Windows Setup."""
from __future__ import annotations

from importlib.metadata import version
import sys


def check_full_profile(backend):
    from PIL import Image
    import numpy
    from ortools.constraint_solver import pywrapcp
    from ortools.sat.python import cp_model
    from pysat.solvers import Solver
    from wgpu.backends import wgpu_native

    solver = pywrapcp.Solver("setup")
    x = solver.IntVar(0, 1, "x")
    solver.Add(x == 1)
    if not solver.Solve(solver.Phase([x], solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)):
        raise RuntimeError("CP did not solve x = 1.")
    print("PASS CP")

    model = cp_model.CpModel()
    model.add(model.new_bool_var("x") == 1)
    if cp_model.CpSolver().solve(model) != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT did not solve x = 1.")
    print("PASS CP-SAT")

    with Solver(name="glucose4", bootstrap_with=[[1]]) as sat:
        if not sat.solve() or sat.solve(assumptions=[-1]):
            raise RuntimeError("Glucose failed its SAT/UNSAT control.")
    print("PASS SAT (Glucose 4)")
    print("PASS WebGPU native library, NumPy and Pillow")

    if backend in ('auto', 'cuda'):
        import cupy
        # A machine without an NVIDIA driver must still receive every CUDA wheel.
        for package in ('cupy-cuda12x', 'nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12'):
            print(f"PASS {package} {version(package)}")


if __name__ == "__main__":
    try:
        check_full_profile(sys.argv[1])
    except Exception as exc:
        sys.exit(f"FAIL Windows full profile: {exc}")
