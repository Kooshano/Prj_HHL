"""
QLS - Quantum Linear Solver Package

This package provides implementations of quantum algorithms for solving linear systems of equations,
specifically the Harrow-Hassidim-Lloyd (HHL) algorithm.

Main Components:
    - HHL: Quantum algorithm implementation for solving linear systems
    - NumPyLinearSolver: Classical reference implementation using NumPy
    - LinearSolver: Abstract base class for linear system solvers
    - Matrices: Matrix representations for quantum circuits
    - Observables: Quantum observables for extracting information from solutions
    - GPU Utils: Utilities for GPU-accelerated quantum simulations

Example Usage:
    >>> import numpy as np
    >>> from QLS.hhl import HHL
    >>> from QLS.gpu_utils import create_gpu_backend
    >>> 
    >>> # Create a simple 2x2 system: Ax = b
    >>> A = np.array([[1, 0.5], [0.5, 1]])
    >>> b = np.array([1, 0])
    >>> 
    >>> # Create HHL solver with GPU backend
    >>> backend, use_gpu = create_gpu_backend()
    >>> hhl = HHL(epsilon=1e-2, quantum_instance=backend)
    >>> 
    >>> # Solve the system
    >>> result = hhl.solve(A, b)
    >>> solution = result.state
"""

__version__ = "1.0.0"

# Main exports
from QLS.hhl import HHL
from QLS.numpy_linear_solver import NumPyLinearSolver
from QLS.linear_solver import LinearSolver, LinearSolverResult

__all__ = [
    'HHL',
    'NumPyLinearSolver',
    'LinearSolver',
    'LinearSolverResult',
]
