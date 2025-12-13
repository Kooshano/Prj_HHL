#!/usr/bin/env python
"""Complete HHL Demo - Optimized for quick demonstration"""

import numpy as np
from scipy.sparse import coo_matrix
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from QLS.numpy_linear_solver import NumPyLinearSolver
from QLS.hhl import HHL
import time

print("="*70)
print("HHL QUANTUM LINEAR SOLVER DEMONSTRATION")
print("="*70)

# ----- Parameters -----
NUM_WORK_QUBITS = 2  # 4×4 matrix (fast demo)
DIM = 2 ** NUM_WORK_QUBITS
make_hermitian = True
target_kappa = 10  # Reduced for stability
density = 0.9
noise_level = 1

print(f"\nProblem Size: {DIM}×{DIM} matrix ({NUM_WORK_QUBITS} qubits)")

# ----- Generate Matrix -----
def generate_sparse_spd(n, kappa, density, noise_level):
    eigs = np.logspace(0, np.log10(kappa), n)
    rows = np.arange(n)
    A = coo_matrix((eigs, (rows, rows)), shape=(n, n))
    
    nnz = int(density * n * n)
    off = max(nnz - n, 0)
    off -= off % 2
    half = off // 2
    
    if half > 0:
        i = np.random.randint(0, n, half * 2)
        j = np.random.randint(0, n, half * 2)
        mask = (i != j)
        i, j = i[mask][:half], j[mask][:half]
        eps = noise_level * eigs.min()
        vals = np.random.uniform(-eps, eps, size=half)
        A = A + coo_matrix((vals, (i, j)), shape=(n, n)) + \
            coo_matrix((vals, (j, i)), shape=(n, n))
    
    return A.tocsr()

if make_hermitian:
    A_sparse = generate_sparse_spd(DIM, target_kappa, density, noise_level)
    A = A_sparse.toarray()
else:
    M = np.random.randn(DIM, DIM)
    A = (M + M.T) / 2

is_herm = np.allclose(A, A.conj().T, atol=1e-12)
cond_A = np.linalg.cond(A)
nnz = np.count_nonzero(A)

print(f"Hermitian: {is_herm}, Condition Number: {cond_A:.3e}")
print(f"Sparsity: {nnz}/{DIM*DIM} = {nnz/(DIM*DIM):.1%}")

# ----- Setup -----
b_vec = np.zeros(DIM, dtype=float)
b_vec[0] = 1

print("\n" + "-"*70)
print("CLASSICAL SOLUTION")
print("-"*70)
classical_res = NumPyLinearSolver().solve(A, b_vec / np.linalg.norm(b_vec))
print(f"Solution vector: {classical_res.state}")
print(f"Euclidean norm: {classical_res.euclidean_norm:.6f}")

print("\n" + "-"*70)
print("QUANTUM (HHL) SOLUTION")
print("-"*70)

# Setup backend
backend = AerSimulator(method='statevector', device='CPU', precision='double')
print(f"Backend: {backend.name}")
print(f"Available devices: {backend.available_devices()}")

# Solve with HHL
start_time = time.time()
hhl_solver = HHL(epsilon=1e-2, quantum_instance=backend)
quantum_res = hhl_solver.solve(A, b_vec)
elapsed = time.time() - start_time

# Extract solution
sv = Statevector(quantum_res.state).data
total_qubits = int(np.log2(len(sv)))
base_index = 1 << (total_qubits - 1)
x_quantum = np.array([sv[base_index + i] for i in range(DIM)])
x_quantum = quantum_res.euclidean_norm * x_quantum / np.linalg.norm(x_quantum)

print(f"Solution vector: {x_quantum.real}")
print(f"Euclidean norm: {quantum_res.euclidean_norm:.6f}")
print(f"Time: {elapsed:.3f} seconds")

print("\n" + "-"*70)
print("COMPARISON")
print("-"*70)
x_classical = classical_res.state
error = np.linalg.norm(x_classical - x_quantum)
print(f"Classical: {x_classical}")
print(f"Quantum:   {x_quantum.real}")
print(f"L2 error:  {error:.6f}")
print(f"Relative error: {error/np.linalg.norm(x_classical)*100:.2f}%")

print("\n" + "="*70)
if error < 0.1:
    print("✓✓✓ SUCCESS: HHL working correctly! ✓✓✓")
else:
    print(f"⚠ Warning: Error ({error:.4f}) higher than expected")
print("="*70)


