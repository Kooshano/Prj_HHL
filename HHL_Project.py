# %%
import numpy as np
from scipy.sparse import coo_matrix
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from QLS.numpy_linear_solver import NumPyLinearSolver   # classical
from QLS.hhl import HHL              # quantum HHL
import time

# %%
# ----- user parameters ------------------------------------------------------
make_hermitian = True       # True → Hermitian SPD; False → general non-Hermitian
target_kappa   = 1e3        # desired condition number κ(A)
density        = 0.4        # fraction of nonzero entries (0<density≤1)
noise_level    = 1e1       # relative off-diag noise for SPD case

# %%
# ----- problem definition ---------------------------------------------------
NUM_WORK_QUBITS = 4                
DIM             = 2 ** NUM_WORK_QUBITS

# %%
# ----- helpers ---------------------------------------------------------------
def generate_sparse_spd(n, kappa, density, noise_level):
    """Hermitian SPD with log-spaced eigenvalues and random sparse off-diag noise."""
    # 1) log-spaced eigenvalues
    eigs = np.logspace(0, np.log10(kappa), n)
    # 2) diagonal entries
    rows = np.arange(n); cols = rows; data = eigs
    A = coo_matrix((data, (rows, cols)), shape=(n, n))
    # 3) add symmetric off-diagonal noise
    total = n*n
    nnz   = int(density*total)
    off   = max(nnz - n, 0)
    off  -= off % 2
    half  = off//2
    if half>0:
        i = np.random.randint(0,n,half*2)
        j = np.random.randint(0,n,half*2)
        mask = (i!=j)
        i,j = i[mask][:half], j[mask][:half]
        eps  = noise_level * eigs.min()
        vals = np.random.uniform(-eps, eps, size=half)
        A   = A + coo_matrix((vals,(i,j)),shape=(n,n))\
                + coo_matrix((vals,(j,i)),shape=(n,n))
    return A.tocsr()

# %%
def generate_general(n, kappa, density):
    """General (non-Hermitian) matrix with approx κ via SVD, then sparsified."""
    # 1) U, V random orthonormal
    U,_  = np.linalg.qr(np.random.randn(n,n))
    V,_  = np.linalg.qr(np.random.randn(n,n))
    # 2) singular values
    s    = np.logspace(0, np.log10(kappa), n)
    A    = U @ np.diag(s) @ V.T
    # 3) sparsify by zeroing random entries
    mask = (np.random.rand(n,n) < density)
    A   *= mask
    return A

# %%
# ----- build A ----------------------------------------------------------------
if make_hermitian:
    A_sparse = generate_sparse_spd(DIM, target_kappa, density, noise_level)
    A        = A_sparse.toarray()
else:
    A = generate_general(DIM, target_kappa, density)


# %%
# ----- checks & print ---------------------------------------------------------
is_herm = np.allclose(A, A.conj().T, atol=1e-12)
cond_A  = np.linalg.cond(A)
nnz     = np.count_nonzero(A)

print("A =\n", A)   # uncomment to see the full matrix


print(f"A (dim={DIM}×{DIM}), Hermitian? {is_herm}, κ(A) ≈ {cond_A:.3e}, "
      f"sparsity={nnz}/{DIM*DIM}={nnz/(DIM*DIM):.2%}\n")
# %%
# ----- right-hand side -------------------------------------------------------
b_vec = np.zeros(DIM, dtype=complex if np.iscomplexobj(A) else float)
b_vec[0] = 1
b_vec

# %%
# ----- classical solution ----------------------------------------------------
classical_res = NumPyLinearSolver().solve(
    A,
    b_vec / np.linalg.norm(b_vec)
)

# %%
# ----- quantum (HHL) solution ------------------------------------------------
# calculate the time for the quantum solver
start_time = time.time()


backend = AerSimulator(method='statevector',
                           device='GPU',
                           precision='double') 
print(backend.available_devices())
hhl_solver  = HHL(epsilon=1e-3, quantum_instance=backend)
quantum_res = hhl_solver.solve(A, b_vec)

def extract_solution(result, n_work_qubits: int) -> np.ndarray:
    sv           = Statevector(result.state).data
    total_qubits = int(np.log2(len(sv)))
    base_index   = 1 << (total_qubits - 1)
    amps         = np.array([sv[base_index + i]
                             for i in range(2 ** n_work_qubits)])
    return result.euclidean_norm * amps / np.linalg.norm(amps)

x_classical = classical_res.state
x_quantum   = extract_solution(quantum_res, NUM_WORK_QUBITS)
# calculate the time taken by the quantum solver
end_time = time.time()
# print time taken
print(f"Quantum solver took {end_time - start_time:.3f} seconds.")

# %%
# ----- results ---------------------------------------------------------------
print("Classical solution vector:", x_classical)
print("Quantum   solution vector:", x_quantum, "\n")
print("Classical Euclidean norm:", classical_res.euclidean_norm)
print("Quantum   Euclidean norm:", quantum_res.euclidean_norm, "\n")
print("‖x_classical − x_quantum‖₂ =",
      np.linalg.norm(x_classical - x_quantum))

# %%
# %% [markdown]
# ## Log run results to JSON (robust to malformed file)

# %%
import os
import json
from datetime import datetime

def serialize_complex_vector(vec):
    return [[float(c.real), float(c.imag)] for c in vec]

record = {
    "timestamp":             datetime.now().isoformat(),
    "dim":                   DIM,
    "make_hermitian":        make_hermitian,
    "is_hermitian":          bool(is_herm),
    "condition_number":      cond_A,
    "nnz":                   nnz,
    "density":               nnz/(DIM*DIM),
    "noise_level":           noise_level if make_hermitian else None,
    "time_quantum_sec":      end_time - start_time,
    "euclid_norm_classical": classical_res.euclidean_norm,
    "euclid_norm_quantum":   quantum_res.euclidean_norm,
    "diff_norm":             float(np.linalg.norm(x_classical - x_quantum)),
    "x_classical":           serialize_complex_vector(x_classical),
    "x_quantum":             serialize_complex_vector(x_quantum),
    "matrix":                A.tolist(),
}

logfile = "hhl_runs_log.json"

# Try to load existing data; if it fails or is malformed, overwrite
try:
    with open(logfile, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON is not a list")
except (FileNotFoundError, json.JSONDecodeError, ValueError):
    data = []

data.append(record)

with open(logfile, "w") as f:
    json.dump(data, f, indent=2)

print(f"Logged run to {logfile} (total runs: {len(data)})")

# %%



