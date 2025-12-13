#!/usr/bin/env python
"""Quick test to verify HHL works correctly."""

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from QLS.numpy_linear_solver import NumPyLinearSolver
from QLS.hhl import HHL
import time

print("="*60)
print("QUICK HHL VERIFICATION TEST")
print("="*60)

# Test with smaller 4x4 problem (2 qubits)
DIM = 4
print(f"\nTesting with {DIM}×{DIM} matrix (fast)...")

# Create a simple Hermitian matrix
np.random.seed(42)
M = np.random.randn(DIM, DIM)
A = (M + M.T) / 2  # Make Hermitian
A = A + np.eye(DIM) * 5  # Make well-conditioned

b_vec = np.zeros(DIM)
b_vec[0] = 1

print(f"Condition number: {np.linalg.cond(A):.2f}")

# Classical solution
print("\n1. Classical solve...")
classical_res = NumPyLinearSolver().solve(A, b_vec / np.linalg.norm(b_vec))
print(f"   ✓ Classical norm: {classical_res.euclidean_norm:.6f}")

# Quantum solution
print("\n2. Quantum (HHL) solve...")
backend = AerSimulator(method='statevector', device='CPU')
start = time.time()
hhl_solver = HHL(epsilon=1e-2, quantum_instance=backend)
quantum_res = hhl_solver.solve(A, b_vec)
elapsed = time.time() - start

# Extract solution
sv = Statevector(quantum_res.state).data
total_qubits = int(np.log2(len(sv)))
base_index = 1 << (total_qubits - 1)
x_quantum = np.array([sv[base_index + i] for i in range(DIM)])
x_quantum = quantum_res.euclidean_norm * x_quantum / np.linalg.norm(x_quantum)

print(f"   ✓ Quantum norm: {quantum_res.euclidean_norm:.6f}")
print(f"   ✓ Time: {elapsed:.2f} seconds")

# Compare
x_classical = classical_res.state
error = np.linalg.norm(x_classical - x_quantum)
print(f"\n3. Solution comparison:")
print(f"   Classical: {x_classical}")
print(f"   Quantum:   {x_quantum.real}")
print(f"   ✓ Error: {error:.6f}")

print("\n" + "="*60)
if error < 0.1:
    print("✓✓✓ SUCCESS: HHL is working correctly! ✓✓✓")
else:
    print("⚠ Warning: Error is larger than expected")
print("="*60)


