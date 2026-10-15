"""Explicit gates for tests that exercise optional native solver packages.

A present but broken package still fails import; only an absent package is a
reason to skip its real native-engine tests. Pure DFS/native-control and mocked
lifecycle tests do not depend on these gates.
"""
import importlib.util
import unittest

HAS_ORTOOLS = importlib.util.find_spec('ortools') is not None
HAS_SAT = importlib.util.find_spec('pysat') is not None
HAS_CUPY = importlib.util.find_spec('cupy') is not None
if HAS_ORTOOLS:
    from ortools.sat.python import cp_model
else:
    cp_model = None
requires_ortools = unittest.skipUnless(HAS_ORTOOLS, 'Optional OR-Tools is not installed for this interpreter')
requires_solver = unittest.skipUnless(HAS_ORTOOLS or HAS_SAT, 'Optional OR-Tools/python-sat engines are not installed for this interpreter')

requires_sat = unittest.skipUnless(HAS_SAT, 'Optional python-sat is not installed for this interpreter')
requires_cupy = unittest.skipUnless(HAS_CUPY, 'Optional CuPy is not installed for this interpreter')
