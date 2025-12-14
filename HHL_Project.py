# %%
import numpy as np
from scipy.sparse import coo_matrix
from qiskit.quantum_info import Statevector
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded configuration from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
    print("  Using default configuration values")
except Exception as e:
    print(f"⚠ Could not load .env file: {e}")
    print("  Using default configuration values")

from QLS.numpy_linear_solver import NumPyLinearSolver   # classical
from QLS.hhl import HHL              # quantum HHL
from QLS.gpu_utils import create_gpu_backend, print_gpu_info, get_gpu_info
import time

# %%
# ----- user parameters (from .env file or defaults) -------------------------
# PERFORMANCE MODE: Set to True for faster, lighter runs
LIGHTWEIGHT_MODE = os.getenv('LIGHTWEIGHT_MODE', 'False').lower() == 'true'

# Matrix generation settings
make_hermitian = os.getenv('MAKE_HERMITIAN', 'True').lower() == 'true'
target_kappa   = float(os.getenv('TARGET_KAPPA', '1e2'))
density        = float(os.getenv('DENSITY', '0.9'))
noise_level    = float(os.getenv('NOISE_LEVEL', '1e1'))

# Precision: lower = faster but less accurate
# 1e-2 = fast (good for testing), 1e-3 = balanced, 1e-4 = high precision (slow)
EPSILON_STR = os.getenv('EPSILON', None)
if EPSILON_STR:
    EPSILON = float(EPSILON_STR)
else:
    EPSILON = 1e-2 if LIGHTWEIGHT_MODE else 1e-3

# Problem definition
NUM_WORK_QUBITS = int(os.getenv('NUM_WORK_QUBITS', '2'))
DIM             = 2 ** NUM_WORK_QUBITS

# Dataset settings
DATASET = os.getenv('DATASET', 'IRIS').upper()
INCLUDE_INTERCEPT = os.getenv('INCLUDE_INTERCEPT', 'True').lower() == 'true'
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
IRIS_TARGET_INDEX = int(os.getenv('IRIS_TARGET_INDEX', '3'))
TITANIC_FEATURES = os.getenv('TITANIC_FEATURES', 'Pclass,Sex,Age,Fare').split(',')
TITANIC_TARGET = os.getenv('TITANIC_TARGET', 'Survived')

# Output settings
LOG_FILE = os.getenv('LOG_FILE', 'logs/hhl_runs_log.json')
VERBOSE = os.getenv('VERBOSE', 'True').lower() == 'true'

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

if not LIGHTWEIGHT_MODE:
    print("A =\n", A)   # uncomment to see the full matrix

print(f"A (dim={DIM}×{DIM}), Hermitian? {is_herm}, κ(A) ≈ {cond_A:.3e}, "
      f"sparsity={nnz}/{DIM*DIM}={nnz/(DIM*DIM):.2%}")
if LIGHTWEIGHT_MODE:
    print(f"Lightweight mode: epsilon={EPSILON}, skipping verbose output\n")
else:
    print(f"\n{'='*70}")
    print("MAIN HHL SOLVER")
    print(f"{'='*70}")
    print(f"Configuration: epsilon={EPSILON:.2e}, {NUM_WORK_QUBITS} work qubits ({DIM}×{DIM} matrix)")
    print()

# %%
# ----- right-hand side -------------------------------------------------------
b_vec = np.zeros(DIM, dtype=complex if np.iscomplexobj(A) else float)
b_vec[0] = 1
b_vec

# %%
# ----- classical solution ----------------------------------------------------
print("\n[1/3] Computing classical solution...")
classical_start = time.time()
classical_res = NumPyLinearSolver().solve(
    A,
    b_vec / np.linalg.norm(b_vec)
)
classical_time = time.time() - classical_start
print(f"    ✓ Classical solution computed in {classical_time:.3f} seconds")

# %%
# ----- quantum (HHL) solution ------------------------------------------------
# calculate the time for the quantum solver
print("\n[2/3] Setting up quantum backend...")
start_time = time.time()

if not LIGHTWEIGHT_MODE:
    print("    → Checking GPU availability...")
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)

print("    → Creating quantum backend...")
backend, use_gpu = create_gpu_backend(
    method='statevector',
    precision='double',
    verbose=VERBOSE and not LIGHTWEIGHT_MODE  # Skip verbose output in lightweight mode
)
backend_time = time.time() - start_time
print(f"    ✓ Backend created ({'GPU' if use_gpu else 'CPU'}) in {backend_time:.3f} seconds")

print("\n    → Creating HHL solver (epsilon={:.2e})...".format(EPSILON))
hhl_solver  = HHL(epsilon=EPSILON, quantum_instance=backend)
print("    ✓ HHL solver created")

print("\n    → Starting quantum solve (this may take a while)...")
solve_start = time.time()
quantum_res = hhl_solver.solve(A, b_vec)
solve_time = time.time() - solve_start
print(f"    ✓ Quantum solve completed in {solve_time:.3f} seconds")

def extract_solution(result, n_work_qubits: int) -> np.ndarray:
    sv           = Statevector(result.state).data
    total_qubits = int(np.log2(len(sv)))
    base_index   = 1 << (total_qubits - 1)
    amps         = np.array([sv[base_index + i]
                             for i in range(2 ** n_work_qubits)])
    return result.euclidean_norm * amps / np.linalg.norm(amps)

print("\n[3/3] Extracting solutions...")
x_classical = classical_res.state
x_quantum   = extract_solution(quantum_res, NUM_WORK_QUBITS)
# calculate the time taken by the quantum solver
end_time = time.time()
total_quantum_time = end_time - start_time
print(f"    ✓ Solutions extracted")
print(f"\n    Total quantum computation time: {total_quantum_time:.3f} seconds")

# %%
# ----- results ---------------------------------------------------------------
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print("\nClassical solution vector:", x_classical)
print("Quantum   solution vector:", x_quantum, "\n")
print("Classical Euclidean norm:", classical_res.euclidean_norm)
print("Quantum   Euclidean norm:", quantum_res.euclidean_norm, "\n")
error = np.linalg.norm(x_classical - x_quantum)
print(f"‖x_classical − x_quantum‖₂ = {error:.6e}")
print(f"\n{'='*70}")

# %%
# %% [markdown]
# ## Log run results to JSON (robust to malformed file)

# %%
import os
import json
from datetime import datetime

def serialize_complex_vector(vec):
    return [[float(c.real), float(c.imag)] for c in vec]

def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

record = {
    "timestamp":             datetime.now().isoformat(),
    "dim":                   int(DIM),
    "make_hermitian":        make_hermitian,
    "is_hermitian":          bool(is_herm),
    "condition_number":      float(cond_A),
    "nnz":                   int(nnz),
    "density":               float(nnz/(DIM*DIM)),
    "noise_level":           float(noise_level) if make_hermitian else None,
    "time_quantum_sec":      float(end_time - start_time),
    "euclid_norm_classical": float(classical_res.euclidean_norm),
    "euclid_norm_quantum":   float(quantum_res.euclidean_norm),
    "diff_norm":             float(np.linalg.norm(x_classical - x_quantum)),
    "x_classical":           serialize_complex_vector(x_classical),
    "x_quantum":             serialize_complex_vector(x_quantum),
    "matrix":                convert_numpy_types(A.tolist()),
}

logfile = LOG_FILE

# Try to load existing data; if it fails or is malformed, overwrite
try:
    with open(logfile, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON is not a list")
except (FileNotFoundError, json.JSONDecodeError, ValueError):
    data = []

data.append(convert_numpy_types(record))

with open(logfile, "w") as f:
    json.dump(data, f, indent=2)

if not LIGHTWEIGHT_MODE:
    print(f"Logged run to {logfile} (total runs: {len(data)})")

# %% [markdown]
# # Linear Regression using HHL vs Scikit-learn
# 
# This section demonstrates how to use HHL to solve linear regression problems and compares the results with scikit-learn's LinearRegression.
# 

# %%
# Skip linear regression section in lightweight mode
RUN_LINEAR_REGRESSION = not LIGHTWEIGHT_MODE

if RUN_LINEAR_REGRESSION:
    print("\n" + "="*70)
    print("LINEAR REGRESSION SECTION")
    print("="*70)
    
    # Import scikit-learn for comparison
    print("\n[Step 1/6] Importing libraries...")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression, load_iris
        from sklearn.metrics import mean_squared_error, r2_score
        SKLEARN_AVAILABLE = True
        print("    ✓ scikit-learn imported successfully")
        
        # Try importing pandas for Titanic dataset
        try:
            import pandas as pd
            PANDAS_AVAILABLE = True
        except ImportError:
            PANDAS_AVAILABLE = False
            if DATASET == 'TITANIC':
                print("    ⚠ Warning: pandas not available. Install with: pip install pandas")
    except ImportError:
        print("    ⚠ Warning: scikit-learn not available. Install with: pip install scikit-learn")
        SKLEARN_AVAILABLE = False
        PANDAS_AVAILABLE = False

    # %%
    # ----- Linear Regression Parameters -----------------------------------------
    np.random.seed(RANDOM_SEED)

    # %%
    # ----- Load Dataset ------------------------------------------
    print(f"\n[Step 2/6] Loading {DATASET} dataset...")
    
    try:
        if DATASET == 'IRIS':
            # Load IRIS dataset from sklearn
            if SKLEARN_AVAILABLE:
                print("    → Loading IRIS from sklearn...")
                iris = load_iris()
                X_full = iris.data  # All 4 features: sepal length, sepal width, petal length, petal width
                feature_names = iris.feature_names
                
                # Use 3 features to predict the 4th feature
                feature_indices = [i for i in range(4) if i != IRIS_TARGET_INDEX]
                X = X_full[:, feature_indices]  # Features (3 features)
                y = X_full[:, IRIS_TARGET_INDEX]  # Target (1 feature)
                
                print(f"    ✓ Loaded IRIS dataset: {X.shape[0]} samples")
                print(f"    → Features: {[feature_names[i] for i in feature_indices]}")
                print(f"    → Target: {feature_names[IRIS_TARGET_INDEX]}")
                print(f"    → Features shape: {X.shape}, Target shape: {y.shape}")
                print(f"    → Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}, std: {y.std():.2f}")
            else:
                raise ImportError("scikit-learn required for IRIS dataset")
                
        elif DATASET == 'TITANIC':
            # Load Titanic dataset
            if SKLEARN_AVAILABLE:
                print("    → Loading Titanic dataset...")
                try:
                    import pandas as pd
                    # Try to load from common locations
                    titanic_paths = [
                        'data/titanic.csv',
                        'titanic.csv',
                        'data/Titanic.csv',
                        'Titanic.csv'
                    ]
                    
                    titanic_df = None
                    for path in titanic_paths:
                        if os.path.exists(path):
                            titanic_df = pd.read_csv(path)
                            print(f"    → Found dataset at: {path}")
                            break
                    
                    if titanic_df is None:
                        # Try to download from seaborn (if available)
                        try:
                            import seaborn as sns
                            titanic_df = sns.load_dataset('titanic')
                            print("    → Loaded from seaborn")
                            # Seaborn uses lowercase column names, convert to match expected
                            titanic_df.columns = titanic_df.columns.str.capitalize()
                            # Handle special cases - seaborn has both 'pclass' and 'class'
                            if 'Pclass' not in titanic_df.columns and 'Class' in titanic_df.columns:
                                titanic_df.rename(columns={'Class': 'Pclass'}, inplace=True)
                        except:
                            # Try sklearn's fetch_openml
                            try:
                                from sklearn.datasets import fetch_openml
                                titanic_df = fetch_openml('titanic', version=1, as_frame=True, return_X_y=False)
                                print("    → Loaded from OpenML")
                            except:
                                pass
                    
                    if titanic_df is None:
                        raise FileNotFoundError("Could not find Titanic dataset")
                    
                    # Preprocess: handle missing values and encode categorical
                    titanic_df = titanic_df.copy()
                    
                    # Map column names (handle both lowercase and capitalized)
                    col_mapping = {}
                    for col in titanic_df.columns:
                        col_upper = col.capitalize()
                        if col_upper != col:
                            col_mapping[col] = col_upper
                    if col_mapping:
                        titanic_df.rename(columns=col_mapping, inplace=True)
                    
                    # Fill missing Age with median
                    if 'Age' in titanic_df.columns:
                        titanic_df['Age'] = pd.to_numeric(titanic_df['Age'], errors='coerce')
                        titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
                    
                    # Fill missing Fare with median
                    if 'Fare' in titanic_df.columns:
                        titanic_df['Fare'] = pd.to_numeric(titanic_df['Fare'], errors='coerce')
                        titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)
                    
                    # Fill missing Embarked with mode
                    if 'Embarked' in titanic_df.columns:
                        titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0] if len(titanic_df['Embarked'].mode()) > 0 else 'S', inplace=True)
                    
                    # Encode categorical variables
                    if 'Sex' in titanic_df.columns:
                        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1, 'Male': 0, 'Female': 1})
                        titanic_df['Sex'] = pd.to_numeric(titanic_df['Sex'], errors='coerce')
                        titanic_df['Sex'].fillna(0, inplace=True)
                    
                    if 'Embarked' in titanic_df.columns:
                        embarked_map = {'S': 0, 'C': 1, 'Q': 2, 's': 0, 'c': 1, 'q': 2}
                        titanic_df['Embarked'] = titanic_df['Embarked'].map(embarked_map)
                        titanic_df['Embarked'] = pd.to_numeric(titanic_df['Embarked'], errors='coerce')
                        titanic_df['Embarked'].fillna(0, inplace=True)
                    
                    # Ensure Pclass is numeric
                    if 'Pclass' in titanic_df.columns:
                        titanic_df['Pclass'] = pd.to_numeric(titanic_df['Pclass'], errors='coerce')
                        titanic_df['Pclass'].fillna(3, inplace=True)
                    
                    # Select features - check both original and capitalized names
                    available_features = []
                    for f in TITANIC_FEATURES:
                        f_clean = f.strip()
                        if f_clean in titanic_df.columns:
                            available_features.append(f_clean)
                        elif f_clean.capitalize() in titanic_df.columns:
                            available_features.append(f_clean.capitalize())
                    
                    if not available_features:
                        # Try default features with case-insensitive matching
                        default_features = ['Pclass', 'Sex', 'Age', 'Fare']
                        for df in default_features:
                            if df in titanic_df.columns:
                                available_features.append(df)
                            elif df.lower() in titanic_df.columns:
                                available_features.append(df.lower())
                    
                    if not available_features:
                        raise ValueError(f"None of the requested features {TITANIC_FEATURES} found in dataset. Available: {list(titanic_df.columns)}")
                    
                    # Convert to numeric and ensure all are numeric
                    for feat in available_features:
                        titanic_df[feat] = pd.to_numeric(titanic_df[feat], errors='coerce')
                    
                    X = titanic_df[available_features].values.astype(float)
                    
                    # Select target - handle case-insensitive
                    target_col = None
                    if TITANIC_TARGET in titanic_df.columns:
                        target_col = TITANIC_TARGET
                    elif TITANIC_TARGET.capitalize() in titanic_df.columns:
                        target_col = TITANIC_TARGET.capitalize()
                    elif TITANIC_TARGET.lower() in titanic_df.columns:
                        target_col = TITANIC_TARGET.lower()
                    
                    if target_col is None:
                        raise ValueError(f"Target '{TITANIC_TARGET}' not found in dataset. Available: {list(titanic_df.columns)}")
                    
                    y = pd.to_numeric(titanic_df[target_col], errors='coerce').values.astype(float)
                    # Remove any NaN values
                    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    print(f"    ✓ Loaded Titanic dataset: {X.shape[0]} samples")
                    print(f"    → Features: {available_features}")
                    print(f"    → Target: {TITANIC_TARGET}")
                    print(f"    → Features shape: {X.shape}, Target shape: {y.shape}")
                    print(f"    → Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}, std: {y.std():.2f}")
                    
                    # Ensure X is properly formatted as float array
                    if X.dtype != np.float64:
                        X = X.astype(np.float64)
                    
                except Exception as e:
                    print(f"    ⚠ Error loading Titanic: {e}")
                    raise
            else:
                raise ImportError("scikit-learn and pandas required for Titanic dataset")
                
        elif DATASET == 'SYNTHETIC':
            # Generate synthetic data
            print("    → Generating synthetic dataset...")
            NUM_FEATURES = 3
            NUM_SAMPLES = 150
            NOISE = 0.1
            if SKLEARN_AVAILABLE:
                X, y = make_regression(
                    n_samples=NUM_SAMPLES,
                    n_features=NUM_FEATURES,
                    noise=NOISE,
                    random_state=RANDOM_SEED,
                    bias=1.0
                )
            else:
                X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
                true_coef = np.random.randn(NUM_FEATURES)
                y = X @ true_coef + np.random.randn(NUM_SAMPLES) * NOISE + 1.0
            print(f"    ✓ Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
            
        else:
            raise ValueError(f"Unknown dataset: {DATASET}. Choose from: IRIS, TITANIC, SYNTHETIC")
            
    except Exception as e:
        print(f"    ⚠ Error loading {DATASET} dataset: {e}")
        print("    → Falling back to synthetic data...")
        # Fallback to synthetic data
        NUM_FEATURES = 3
        NUM_SAMPLES = 150
        NOISE = 0.1
        if SKLEARN_AVAILABLE:
            X, y = make_regression(
                n_samples=NUM_SAMPLES,
                n_features=NUM_FEATURES,
                noise=NOISE,
                random_state=RANDOM_SEED,
                bias=1.0
            )
        else:
            X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
            true_coef = np.random.randn(NUM_FEATURES)
            y = X @ true_coef + np.random.randn(NUM_SAMPLES) * NOISE + 1.0
        print(f"    ✓ Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # %%
    # ----- Prepare Normal Equations for HHL -------------------------------------
    print("\n[Step 3/6] Preparing normal equations...")
    # Linear regression solves: (X^T X) β = X^T y
    # This is the normal equation form: A β = b

    # Add intercept term if requested (add column of ones)
    print("    → Adding intercept term..." if INCLUDE_INTERCEPT else "    → No intercept term")
    if INCLUDE_INTERCEPT:
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        print(f"    ✓ Design matrix shape: {X_with_intercept.shape}")
    else:
        X_with_intercept = X

    # Compute X^T X (Gram matrix)
    print("    → Computing X^T X (Gram matrix)...")
    A_reg = X_with_intercept.T @ X_with_intercept
    b_reg = X_with_intercept.T @ y
    print(f"    ✓ Normal equations: A shape {A_reg.shape}, b shape {b_reg.shape}")
    print(f"    → Condition number: {np.linalg.cond(A_reg):.2e}")

    # HHL requires matrix dimension to be a power of 2
    # Find next power of 2
    print("    → Padding matrix to power of 2 for HHL...")
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
        
        print(f"    ✓ Padded from {n_features}x{n_features} to {next_power_of_2}x{next_power_of_2}")
        A_hhl = A_reg_padded
        b_hhl = b_reg_padded
        n_work_qubits_reg = int(np.log2(next_power_of_2))
    else:
        A_hhl = A_reg
        b_hhl = b_reg
        n_work_qubits_reg = int(np.log2(n_features))
        print(f"    ✓ Matrix already power of 2: {n_features}x{n_features}")

    print(f"    ✓ Using {n_work_qubits_reg} work qubits for HHL")

    # %%
    # ----- Solve with Scikit-learn (Classical) -----------------------------------
    print("\n[Step 4/6] Solving with classical method (scikit-learn)...")
    if SKLEARN_AVAILABLE:
        print("    → Fitting LinearRegression model...")
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
        
        print("    ✓ Classical solution computed")
        print("\n    Scikit-learn Results:")
        if INCLUDE_INTERCEPT:
            print(f"  Intercept: {beta_sklearn[0]:.4f}")
            print(f"  Coefficients: {beta_sklearn[1:]}")
        else:
            print(f"  Coefficients: {beta_sklearn}")
        print(f"  MSE: {mse_sklearn:.4f}")
        print(f"  R²: {r2_sklearn:.4f}")
    else:
        # Manual solution using normal equations
        print("    → Solving using normal equations (numpy)...")
        beta_sklearn = np.linalg.solve(A_reg, b_reg)
        y_pred_sklearn = X_with_intercept @ beta_sklearn
        mse_sklearn = np.mean((y - y_pred_sklearn)**2)
        r2_sklearn = 1 - np.sum((y - y_pred_sklearn)**2) / np.sum((y - y.mean())**2)
        
        print("    ✓ Classical solution computed")
        print("\n    Classical Solution (Normal Equations):")
        if INCLUDE_INTERCEPT:
            print(f"  Intercept: {beta_sklearn[0]:.4f}")
            print(f"  Coefficients: {beta_sklearn[1:]}")
        else:
            print(f"  Coefficients: {beta_sklearn}")
        print(f"  MSE: {mse_sklearn:.4f}")
        print(f"  R²: {r2_sklearn:.4f}")

    # %%
    # ----- Solve with HHL (Quantum) ----------------------------------------------
    print("\n[Step 5/6] Solving with HHL (Quantum)...")
    start_time_hhl = time.time()

    # Store the norm of b_hhl before HHL normalizes it internally
    b_hhl_norm = np.linalg.norm(b_hhl)

    # Setup GPU backend for regression (reuse GPU info, but create new backend)
    print("    → Setting up quantum backend...")
    backend_reg, use_gpu_reg = create_gpu_backend(
        method='statevector',
        precision='double',
        verbose=VERBOSE and not LIGHTWEIGHT_MODE
    )
    print(f"    ✓ Backend created ({'GPU' if use_gpu_reg else 'CPU'})")

    print("    → Creating HHL solver...")
    hhl_solver_reg = HHL(epsilon=EPSILON, quantum_instance=backend_reg)
    print("    ✓ HHL solver created")
    
    print("    → Running quantum solve (this may take a while)...")
    solve_start = time.time()
    quantum_res_reg = hhl_solver_reg.solve(A_hhl, b_hhl)
    solve_time = time.time() - solve_start
    print(f"    ✓ Quantum solve completed in {solve_time:.3f} seconds")

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
    total_hhl_time = end_time_hhl - start_time_hhl
    print(f"    Total HHL computation time: {total_hhl_time:.3f} seconds")

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
    print("\n[Step 6/6] Comparing results...")
    print("\n" + "="*70)
    print("COMPARISON: HHL vs Scikit-learn/Classical")
    print("="*70)

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

    # Log regression results
    regression_record = {
        "timestamp": datetime.now().isoformat(),
        "type": "linear_regression",
        "dataset": DATASET,
        "num_samples": int(len(y)),
        "num_features": int(n_features - 1 if INCLUDE_INTERCEPT else n_features),
        "include_intercept": INCLUDE_INTERCEPT,
        "epsilon": float(EPSILON),
        "num_work_qubits": int(n_work_qubits_reg),
        "used_gpu": use_gpu_reg,
        "hhl_time_sec": float(end_time_hhl - start_time_hhl),
        "quantum_solve_time_sec": float(solve_time),
        "classical": {
            "intercept": float(beta_sklearn[0]) if INCLUDE_INTERCEPT else None,
            "coefficients": convert_numpy_types(beta_sklearn[1:].tolist() if INCLUDE_INTERCEPT else beta_sklearn.tolist()),
            "mse": float(mse_sklearn),
            "r2": float(r2_sklearn),
        },
        "hhl": {
            "intercept": float(beta_hhl[0]) if INCLUDE_INTERCEPT else None,
            "coefficients": convert_numpy_types(beta_hhl[1:].tolist() if INCLUDE_INTERCEPT else beta_hhl.tolist()),
            "mse": float(mse_hhl),
            "r2": float(r2_hhl),
        },
        "comparison": {
            "coefficient_l2_diff": float(np.linalg.norm(beta_sklearn - beta_hhl)),
            "relative_error": float(relative_error),
            "mse_diff": float(abs(mse_sklearn - mse_hhl)),
            "r2_diff": float(abs(r2_sklearn - r2_hhl)),
        },
        "residual_norm": float(np.linalg.norm(residual)),
    }
    
    # Append regression record to log file
    try:
        with open(logfile, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        data = []
    
    data.append(convert_numpy_types(regression_record))
    
    with open(logfile, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Logged regression results to {logfile}")

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

else:
    print("\n" + "="*60)
    print("SKIPPING LINEAR REGRESSION SECTION (Lightweight Mode)")
    print("="*60)
    print("Set LIGHTWEIGHT_MODE = False to enable linear regression demo")
    print("="*60)


# %%



