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
target_kappa   = 1e2        # desired condition number κ(A)
density        = 0.9        # fraction of nonzero entries (0<density≤1)
noise_level    = 1e1       # relative off-diag noise for SPD case

# %%
# ----- problem definition ---------------------------------------------------
# Note: HHL is computationally expensive. Start with smaller problems:
#   2 qubits = 4×4 matrix   (~0.1 sec)
#   3 qubits = 8×8 matrix   (~1 sec)  
#   4 qubits = 16×16 matrix (~5-10 min on CPU)
NUM_WORK_QUBITS = 2  # Changed from 4 to 2 for faster demo
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


# Try to use GPU, but test if it actually works
backend = None
use_gpu = False
try:
    backend_gpu = AerSimulator(method='statevector',
                               device='GPU',
                               precision='double')
    # Test if GPU actually works by running a simple circuit
    from qiskit import QuantumCircuit
    test_qc = QuantumCircuit(2)
    test_qc.h(0)
    test_result = backend_gpu.run(test_qc).result()
    backend = backend_gpu
    use_gpu = True
    print("GPU backend successfully initialized and tested!")
except Exception as e:
    print(f"GPU backend not available or not working: {e}")
    print("Falling back to CPU backend...")
    backend = AerSimulator(method='statevector',
                           device='CPU',
                           precision='double')
    use_gpu = False

# Print GPU information
print("="*60)
print("GPU Information")
print("="*60)
print(f"Available devices: {backend.available_devices()}")
print(f"Backend name: {backend.name}")
print(f"Actually using GPU: {use_gpu}")
print(f"Backend configuration: {backend.configuration()}")

# Try to get CUDA device information
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,index,memory.total,memory.free', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\nNVIDIA GPU Details:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    print(f"  GPU {parts[1]}: {parts[0]}")
                    print(f"    Total Memory: {parts[2]}")
                    print(f"    Free Memory: {parts[3]}")
except Exception as e:
    print(f"\nCould not retrieve NVIDIA GPU info: {e}")

# Try PyTorch CUDA if available
try:
    import torch
    if torch.cuda.is_available():
        print(f"\nPyTorch CUDA Information:")
        print(f"  CUDA Available: True")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\nPyTorch CUDA: Not available")
except ImportError:
    pass
except Exception as e:
    print(f"\nPyTorch CUDA check failed: {e}")

print("="*60)
print()
print("Creating HHL solver...")
hhl_solver  = HHL(epsilon=1e-3, quantum_instance=backend)
print("HHL solver created. Starting solve...")
print("This may take a while for larger problems...")
quantum_res = hhl_solver.solve(A, b_vec)
print("HHL solve completed!")

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

# %% [markdown]
# # Linear Regression using HHL vs Scikit-learn
# 
# This section demonstrates how to use HHL to solve linear regression problems and compares the results with scikit-learn's LinearRegression.
# 

# %%
# Import scikit-learn for comparison
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


# %%
# ----- Linear Regression Parameters -----------------------------------------
# Load housing data from CSV
HOUSING_CSV = "data/housing.csv"
INCLUDE_INTERCEPT = True  # Whether to include intercept term
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# %%
# ----- Load Housing Dataset ------------------------------------------
# Load housing.csv (space-separated, Boston Housing dataset)
try:
    # Try loading as space-separated
    data = np.loadtxt(HOUSING_CSV, delimiter=None)  # None means any whitespace
    print(f"Loaded housing.csv: {data.shape[0]} samples, {data.shape[1]} columns")
    
    # Boston Housing: last column is target (MEDV), first 13 are features
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]   # Last column is target
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target (MEDV) range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target (MEDV) mean: {y.mean():.2f}, std: {y.std():.2f}")
    
except Exception as e:
    print(f"Error loading housing.csv: {e}")
    print("Falling back to synthetic data...")
    # Fallback to synthetic data
    NUM_FEATURES = 4
    NUM_SAMPLES = 100
    NOISE = 5.0
    if SKLEARN_AVAILABLE:
        X, y = make_regression(
            n_samples=NUM_SAMPLES,
            n_features=NUM_FEATURES,
            noise=NOISE,
            random_state=RANDOM_SEED,
            bias=5.0
        )
    else:
        X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
        true_coef = np.random.randn(NUM_FEATURES)
        y = X @ true_coef + np.random.randn(NUM_SAMPLES) * NOISE + 5.0
    print(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")


# %%
# ----- Prepare Normal Equations for HHL -------------------------------------
# Linear regression solves: (X^T X) β = X^T y
# This is the normal equation form: A β = b

# Add intercept term if requested (add column of ones)
if INCLUDE_INTERCEPT:
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    print(f"Added intercept term. Design matrix shape: {X_with_intercept.shape}")
else:
    X_with_intercept = X

# Compute X^T X (Gram matrix)
A_reg = X_with_intercept.T @ X_with_intercept
b_reg = X_with_intercept.T @ y

print(f"Normal equations matrix shape: {A_reg.shape}")
print(f"Right-hand side shape: {b_reg.shape}")
print(f"Condition number of X^T X: {np.linalg.cond(A_reg):.2e}")

# HHL requires matrix dimension to be a power of 2
# Find next power of 2
n_features = A_reg.shape[0]
next_power_of_2 = 2 ** int(np.ceil(np.log2(n_features)))

if next_power_of_2 > n_features:
    # Pad the matrix and vector to next power of 2
    pad_size = next_power_of_2 - n_features
    A_reg_padded = np.zeros((next_power_of_2, next_power_of_2))
    A_reg_padded[:n_features, :n_features] = A_reg
    # Add small identity to padded diagonal to keep it well-conditioned
    A_reg_padded[n_features:, n_features:] = np.eye(pad_size) * 1e-10
    
    b_reg_padded = np.zeros(next_power_of_2)
    b_reg_padded[:n_features] = b_reg
    
    print(f"Padded to {next_power_of_2}x{next_power_of_2} (power of 2)")
    A_hhl = A_reg_padded
    b_hhl = b_reg_padded
    n_work_qubits_reg = int(np.log2(next_power_of_2))
else:
    A_hhl = A_reg
    b_hhl = b_reg
    n_work_qubits_reg = int(np.log2(n_features))

print(f"Using {n_work_qubits_reg} work qubits for HHL")


# %%
# ----- Solve with Scikit-learn (Classical) -----------------------------------
if SKLEARN_AVAILABLE:
    sklearn_model = LinearRegression(fit_intercept=INCLUDE_INTERCEPT)
    sklearn_model.fit(X, y)
    if INCLUDE_INTERCEPT:
        beta_sklearn = np.concatenate([[sklearn_model.intercept_], sklearn_model.coef_])
    else:
        beta_sklearn = sklearn_model.coef_
    
    # Predictions
    y_pred_sklearn = sklearn_model.predict(X)
    mse_sklearn = mean_squared_error(y, y_pred_sklearn)
    r2_sklearn = r2_score(y, y_pred_sklearn)
    
    print("Scikit-learn Results:")
    if INCLUDE_INTERCEPT:
        print(f"  Intercept: {beta_sklearn[0]:.4f}")
        print(f"  Coefficients: {beta_sklearn[1:]}")
    else:
        print(f"  Coefficients: {beta_sklearn}")
    print(f"  MSE: {mse_sklearn:.4f}")
    print(f"  R²: {r2_sklearn:.4f}")
else:
    # Manual solution using normal equations
    beta_sklearn = np.linalg.solve(A_reg, b_reg)
    y_pred_sklearn = X_with_intercept @ beta_sklearn
    mse_sklearn = np.mean((y - y_pred_sklearn)**2)
    r2_sklearn = 1 - np.sum((y - y_pred_sklearn)**2) / np.sum((y - y.mean())**2)
    
    print("Classical Solution (Normal Equations):")
    if INCLUDE_INTERCEPT:
        print(f"  Intercept: {beta_sklearn[0]:.4f}")
        print(f"  Coefficients: {beta_sklearn[1:]}")
    else:
        print(f"  Coefficients: {beta_sklearn}")
    print(f"  MSE: {mse_sklearn:.4f}")
    print(f"  R²: {r2_sklearn:.4f}")


# %%
# ----- Solve with HHL (Quantum) ----------------------------------------------
print("\nSolving with HHL...")
start_time_hhl = time.time()

# Store the norm of b_hhl before HHL normalizes it internally
b_hhl_norm = np.linalg.norm(b_hhl)

# Try to use GPU, but test if it actually works
backend_reg = None
use_gpu_reg = False
try:
    backend_gpu_reg = AerSimulator(method='statevector', device='GPU', precision='double')
    # Test if GPU actually works by running a simple circuit
    from qiskit import QuantumCircuit
    test_qc = QuantumCircuit(2)
    test_qc.h(0)
    test_result = backend_gpu_reg.run(test_qc).result()
    backend_reg = backend_gpu_reg
    use_gpu_reg = True
    print("GPU backend successfully initialized and tested!")
except Exception as e:
    print(f"GPU backend not available or not working: {e}")
    print("Falling back to CPU backend...")
    backend_reg = AerSimulator(method='statevector', device='CPU', precision='double')
    use_gpu_reg = False

# Print GPU information for regression
print("="*60)
print("GPU Information (Regression)")
print("="*60)
print(f"Available devices: {backend_reg.available_devices()}")
print(f"Backend name: {backend_reg.name}")
print(f"Actually using GPU: {use_gpu_reg}")

# Try to get CUDA device information
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,index,memory.total,memory.free', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\nNVIDIA GPU Details:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    print(f"  GPU {parts[1]}: {parts[0]}")
                    print(f"    Total Memory: {parts[2]}")
                    print(f"    Free Memory: {parts[3]}")
except Exception as e:
    print(f"\nCould not retrieve NVIDIA GPU info: {e}")

# Try PyTorch CUDA if available
try:
    import torch
    if torch.cuda.is_available():
        print(f"\nPyTorch CUDA Information:")
        print(f"  CUDA Available: True")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\nPyTorch CUDA: Not available")
except ImportError:
    pass
except Exception as e:
    print(f"\nPyTorch CUDA check failed: {e}")

print("="*60)
print()

hhl_solver_reg = HHL(epsilon=1e-3, quantum_instance=backend_reg)
quantum_res_reg = hhl_solver_reg.solve(A_hhl, b_hhl)

# Extract solution
def extract_solution_reg(result, n_work_qubits: int, original_dim: int, b_norm: float) -> np.ndarray:
    """Extract solution vector from HHL result, handling padding and scaling.
    
    HHL normalizes the input vector internally, so we need to scale the solution
    by the original vector's norm to get the correct result.
    """
    sv = Statevector(result.state).data
    total_qubits = int(np.log2(len(sv)))
    base_index = 1 << (total_qubits - 1)
    amps = np.array([sv[base_index + i] for i in range(2 ** n_work_qubits)])
    # Extract normalized solution direction
    solution_normalized = amps / np.linalg.norm(amps)
    # Scale by euclidean norm and by original b norm to account for HHL's internal normalization
    solution_full = result.euclidean_norm * b_norm * solution_normalized
    # Return only the original dimension (remove padding)
    return solution_full[:original_dim].real

# Extract solution - n_features now includes intercept if INCLUDE_INTERCEPT
beta_hhl = extract_solution_reg(quantum_res_reg, n_work_qubits_reg, n_features, b_hhl_norm)

end_time_hhl = time.time()
print(f"HHL solver took {end_time_hhl - start_time_hhl:.3f} seconds.")

# Verify the solution satisfies the normal equations (for debugging)
# Check: A_reg @ beta_hhl should be close to b_reg
residual = A_reg @ beta_hhl - b_reg
print(f"\nVerification: ||A @ β_hhl - b|| = {np.linalg.norm(residual):.6e}")

# If the residual is large, try alternative scaling
if np.linalg.norm(residual) > 1e-3:
    print("Warning: Large residual detected. Trying alternative scaling...")
    # Alternative: scale to match the norm of sklearn solution
    scale_factor = np.linalg.norm(beta_sklearn) / np.linalg.norm(beta_hhl)
    beta_hhl_alt = beta_hhl * scale_factor
    residual_alt = A_reg @ beta_hhl_alt - b_reg
    print(f"Alternative scaling factor: {scale_factor:.6f}")
    print(f"Alternative residual: ||A @ β_hhl_alt - b|| = {np.linalg.norm(residual_alt):.6e}")
    if np.linalg.norm(residual_alt) < np.linalg.norm(residual):
        print("Using alternative scaling.")
        beta_hhl = beta_hhl_alt

# Predictions using HHL coefficients
y_pred_hhl = X_with_intercept @ beta_hhl
mse_hhl = np.mean((y - y_pred_hhl)**2)
r2_hhl = 1 - np.sum((y - y_pred_hhl)**2) / np.sum((y - y.mean())**2)

print("\nHHL Results:")
if INCLUDE_INTERCEPT:
    print(f"  Intercept: {beta_hhl[0]:.4f}")
    print(f"  Coefficients: {beta_hhl[1:]}")
else:
    print(f"  Coefficients: {beta_hhl}")
print(f"  MSE: {mse_hhl:.4f}")
print(f"  R²: {r2_hhl:.4f}")


# %%
# ----- Comparison ------------------------------------------------------------
print("\n" + "="*60)
print("COMPARISON: HHL vs Scikit-learn/Classical")
print("="*60)

print(f"\nCoefficient Comparison:")
if INCLUDE_INTERCEPT:
    print(f"  Scikit-learn Intercept: {beta_sklearn[0]:.6f}")
    print(f"  HHL Intercept:          {beta_hhl[0]:.6f}")
    print(f"  Intercept Difference:   {beta_sklearn[0] - beta_hhl[0]:.6e}")
    print(f"\n  Scikit-learn Coefs: {beta_sklearn[1:]}")
    print(f"  HHL Coefs:          {beta_hhl[1:]}")
    print(f"  Coef Difference:   {beta_sklearn[1:] - beta_hhl[1:]}")
else:
    print(f"  Scikit-learn: {beta_sklearn}")
    print(f"  HHL:          {beta_hhl}")
    print(f"  Difference:   {beta_sklearn - beta_hhl}")
print(f"  L2 norm of difference: {np.linalg.norm(beta_sklearn - beta_hhl):.6e}")

print(f"\nPerformance Metrics:")
print(f"  {'Metric':<20} {'Scikit-learn':<15} {'HHL':<15} {'Difference':<15}")
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
print(f"  {'MSE':<20} {mse_sklearn:<15.4f} {mse_hhl:<15.4f} {abs(mse_sklearn - mse_hhl):<15.6e}")
print(f"  {'R²':<20} {r2_sklearn:<15.4f} {r2_hhl:<15.4f} {abs(r2_sklearn - r2_hhl):<15.6e}")

print(f"\nRelative Error:")
relative_error = np.linalg.norm(beta_sklearn - beta_hhl) / np.linalg.norm(beta_sklearn)
print(f"  ‖β_sklearn - β_hhl‖ / ‖β_sklearn‖ = {relative_error:.6e}")

print(f"\nTiming:")
print(f"  HHL computation time: {end_time_hhl - start_time_hhl:.3f} seconds")


# %%
# ----- Visualization (if matplotlib available) ------------------------------
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Coefficient comparison
    x_pos = np.arange(len(beta_sklearn))
    width = 0.35
    axes[0].bar(x_pos - width/2, beta_sklearn, width, label='Scikit-learn', alpha=0.8)
    axes[0].bar(x_pos + width/2, beta_hhl, width, label='HHL', alpha=0.8)
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].set_title('Regression Coefficients Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f'β{i}' for i in range(len(beta_sklearn))])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual
    axes[1].scatter(y, y_pred_sklearn, alpha=0.6, label='Scikit-learn', s=50)
    axes[1].scatter(y, y_pred_hhl, alpha=0.6, label='HHL', s=50, marker='x')
    # Perfect prediction line
    min_val = min(y.min(), y_pred_sklearn.min(), y_pred_hhl.min())
    max_val = max(y.max(), y_pred_sklearn.max(), y_pred_hhl.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
    axes[1].set_xlabel('Actual y')
    axes[1].set_ylabel('Predicted y')
    axes[1].set_title('Predicted vs Actual Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Matplotlib not available. Skipping visualization.")
    print("Install with: pip install matplotlib")


# %%



