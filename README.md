# HHL Quantum Linear Solver Project

A comprehensive implementation of the **Harrow-Hassidim-Lloyd (HHL)** quantum algorithm for solving linear systems of equations, featuring GPU acceleration support and classical comparison methods.

## 🚀 Features

- **Quantum HHL Algorithm**: Full implementation using Qiskit with support for Hermitian and non-Hermitian matrices
- **GPU Acceleration**: Automatic GPU detection and acceleration with CPU fallback
- **Classical Comparison**: NumPy-based classical solver for benchmarking
- **Linear Regression**: Application of HHL to linear regression problems
- **Comprehensive Logging**: Detailed metrics logging including R², MSE, and performance data
- **Production Ready**: Well-documented, properly structured codebase

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [GPU Support](#gpu-support)
- [Logging System](#logging-system)
- [Documentation](#documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## 🔧 Installation 

### Prerequisites

- Python 3.10 or higher
- CUDA toolkit (optional, for GPU support)
- CMake 3.20+ (for building GPU support)

### Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Prj_HHL

# Create and activate environment
conda env create -f environment.yml
conda activate HHL

# Verify installation
python -c "from QLS.hhl import HHL; print('✓ Installation successful')"
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd Prj_HHL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from QLS.hhl import HHL; print('✓ Installation successful')"
```

### Version Compatibility

**Important**: This project requires:
- **Qiskit**: >=2.0.0, <3.0.0
- **qiskit-aer**: >=0.17.0, <0.18.0
- **Python**: 3.10 or higher

#### Fixing Version Conflicts

If you encounter the error:
```
ImportError: cannot import name 'convert_to_target' from 'qiskit.providers'
```

This indicates a version mismatch. Fix it by:

```bash
# Uninstall incompatible versions
pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu

# Install compatible versions
pip install "qiskit>=2.0.0,<3.0.0" "qiskit-aer>=0.17.0,<0.18.0"
```

#### Checking Your Versions

```bash
python -c "import qiskit; import qiskit_aer; print(f'Qiskit: {qiskit.__version__}'); print(f'qiskit-aer: {qiskit_aer.__version__}')"
```

Expected output:
```
Qiskit: 2.x.x
qiskit-aer: 0.17.x
```

### GPU Support Installation

**Important**: The `qiskit-aer-gpu` package on PyPI (v0.15.1) is incompatible with Qiskit 2.x. To enable GPU support with Qiskit 2.x, you need to build qiskit-aer from source.

#### Quick Build (Using Provided Script)

```bash
# Make sure you're in the HHL conda environment
conda activate HHL

# Run the build script
./scripts/setup_gpu.sh build
```

This script will:
1. Check prerequisites (CUDA, CMake)
2. Clone qiskit-aer repository
3. Build with GPU support
4. Install the built package
5. Verify installation

**Build Time**: 10-30 minutes depending on your system

#### Manual Build

For detailed manual build instructions, see [docs/INSTALL_GPU.md](docs/INSTALL_GPU.md).

#### Verify GPU Support

```bash
# Quick test
python scripts/gpu_test.py

# Detailed diagnostic
python scripts/check_gpu_support.py
```

## 🎯 Quick Start

### Simple Example

```python
import numpy as np
from QLS.hhl import HHL
from QLS.gpu_utils import create_gpu_backend

# Create a simple 2x2 linear system: Ax = b
A = np.array([[1.0, 0.5], [0.5, 1.0]])  # Must be Hermitian
b = np.array([1.0, 0.0])

# Create GPU backend (falls back to CPU if GPU unavailable)
backend, use_gpu = create_gpu_backend()

# Create HHL solver
hhl = HHL(epsilon=1e-2, quantum_instance=backend)

# Solve the system
result = hhl.solve(A, b)
solution = result.state

print(f"Solution: {solution}")
print(f"Euclidean norm: {result.euclidean_norm}")
```

### Run Main Project Script

```bash
# Run the full demonstration
python scripts/HHL_Project.py
```

This will:
1. Generate or load a test matrix
2. Solve using classical method (NumPy)
3. Solve using quantum HHL algorithm
4. Compare results
5. Perform linear regression comparison (if not in lightweight mode)
6. Log all results to `logs/hhl_runs_log.json`

### Run Quick Demo

```bash
# Fast demonstration (smaller problem)
python examples/HHL_demo.py
```

## 📁 Project Structure

```
Prj_HHL/
├── src/                          # Source code
│   ├── QLS/                      # Main quantum linear solver package
│   │   ├── __init__.py           # Package initialization
│   │   ├── hhl.py                # HHL quantum solver implementation
│   │   ├── linear_solver.py      # Base linear solver interface
│   │   ├── numpy_linear_solver.py # Classical NumPy-based solver
│   │   ├── gpu_utils.py          # GPU backend utilities
│   │   ├── matrices/             # Matrix implementations
│   │   │   ├── __init__.py
│   │   │   ├── linear_system_matrix.py
│   │   │   ├── numpy_matrix.py
│   │   │   └── tridiagonal_toeplitz.py
│   │   └── observables/          # Quantum observables
│   │       ├── __init__.py
│   │       ├── absolute_average.py
│   │       ├── linear_system_observable.py
│   │       └── matrix_functional.py
│   └── dataset.py                # Dataset loader module
├── scripts/                      # Executable scripts
│   ├── HHL_Project.py            # Main project script
│   ├── setup_gpu.sh              # GPU setup and testing script
│   ├── check_gpu_support.py      # GPU diagnostic script
│   └── gpu_test.py               # GPU compatibility test
├── notebooks/                    # Jupyter notebooks
│   └── simple_example.ipynb      # Simple example notebook
├── examples/                     # Example scripts
│   └── HHL_demo.py               # Quick demonstration
├── data/                         # Data files (CSV datasets)
│   ├── Fish.csv
│   └── housing.csv
├── logs/                         # Runtime logs (gitignored)
│   └── .gitkeep
├── docs/                         # Documentation
│   └── INSTALL_GPU.md            # GPU installation guide
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── VERSION_FIX.md                # Version compatibility troubleshooting guide
```

## 💡 Usage Examples

### Basic Linear System Solving

```python
from QLS.hhl import HHL
from QLS.numpy_linear_solver import NumPyLinearSolver
from QLS.gpu_utils import create_gpu_backend
import numpy as np

# Problem: Solve Ax = b
A = np.array([[1, 0.5], [0.5, 1]])  # Hermitian matrix
b = np.array([1, 0])

# Classical solution
classical_solver = NumPyLinearSolver()
classical_result = classical_solver.solve(A, b)
print(f"Classical solution: {classical_result.state}")

# Quantum solution
backend, _ = create_gpu_backend()
hhl = HHL(epsilon=1e-2, quantum_instance=backend)
quantum_result = hhl.solve(A, b)
print(f"Quantum solution: {quantum_result.state}")
```

### Linear Regression with HHL

The main script (`scripts/HHL_Project.py`) includes a complete linear regression example comparing HHL with scikit-learn. To run it:

```bash
# Set LIGHTWEIGHT_MODE=False in .env or environment
python scripts/HHL_Project.py
```

### Configuration via Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key configuration options:

```env
# Performance mode (skips heavy sections)
LIGHTWEIGHT_MODE=False

# HHL precision (lower = faster but less accurate)
EPSILON=1e-3

# Problem size
NUM_WORK_QUBITS=2  # 2^2 = 4x4 matrix

# Dataset selection
DATASET=IRIS  # Options: IRIS, HOUSING, TITANIC, SYNTHETIC

# Logging
LOG_FILE=logs/hhl_runs_log.json
VERBOSE=True
```

## 🎮 GPU Support

### Automatic GPU Detection

The project automatically detects and uses GPUs when available:

```python
from QLS.gpu_utils import create_gpu_backend

# Automatically tries GPU, falls back to CPU
backend, use_gpu = create_gpu_backend()
print(f"Using GPU: {use_gpu}")
```

### GPU Requirements

- **NVIDIA GPU** with CUDA support (CUDA 11.2+)
- **Qiskit Aer** with GPU support compiled
- **CUDA toolkit** installed

### Building GPU Support

See [docs/INSTALL_GPU.md](docs/INSTALL_GPU.md) for detailed instructions.

**Quick build**:
```bash
conda activate HHL
./scripts/setup_gpu.sh build
```

### Performance Comparison

| Matrix Size | CPU Time | GPU Time (approx) |
|-------------|----------|-------------------|
| 4×4         | ~0.1s    | ~0.05s            |
| 8×8         | ~1s      | ~0.2s             |
| 16×16       | 5-10 min | ~30s              |
| 32×32       | Hours    | ~5-10 min         |

*Actual times depend on system specifications*

## 📊 Logging System

All runs are logged to `logs/hhl_runs_log.json` with comprehensive metrics:

### Basic Linear System Logs

- Timestamp
- Matrix dimensions and properties
- Condition number
- Computation time
- Classical vs quantum solution comparison
- Euclidean norms
- Solution vectors

### Linear Regression Logs

- Dataset information
- Number of samples and features
- HHL parameters (epsilon, qubits, GPU usage)
- **Performance metrics**: R², MSE for both classical and HHL
- Coefficient comparisons
- Relative errors
- Residual norms

### Accessing Logs

```python
import json

with open('logs/hhl_runs_log.json', 'r') as f:
    logs = json.load(f)

# Filter regression results
regression_logs = [log for log in logs if log.get('type') == 'linear_regression']

# Get latest run
latest = logs[-1]
print(f"R² Score: {latest['hhl']['r2']}")
```

## 📚 Documentation

- **GPU Installation**: [docs/INSTALL_GPU.md](docs/INSTALL_GPU.md) - Detailed guide for building GPU support
- **Version Troubleshooting**: [VERSION_FIX.md](VERSION_FIX.md) - Guide for fixing version compatibility issues
- **Simple Example**: [notebooks/simple_example.ipynb](notebooks/simple_example.ipynb) - Step-by-step tutorial
- **Code Documentation**: All QLS modules include comprehensive docstrings

## 📦 Additional Dependencies

The following are optional but recommended:
- **Jupyter**: For running notebooks (`pip install jupyter` or included in conda environment)
- **matplotlib**: For visualizations (included in requirements)
- **scikit-learn**: For linear regression examples (included in requirements)
- **pandas**: For data loading (included in requirements)

## 🧪 Testing

Test suite is currently under development. For now, you can verify the installation by running:

```bash
# Verify installation
python -c "from QLS.hhl import HHL; print('✓ Installation successful')"

# Run example
python examples/HHL_demo.py
```

## 🔍 Troubleshooting

### Import Errors

1. **Check Python version**: Must be 3.10+
   ```bash
   python --version
   ```

2. **Verify package versions**:
   ```bash
   pip list | grep qiskit
   ```

3. **Reinstall if needed**:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```

### Version Compatibility Issues

If you see:
```
ImportError: cannot import name 'convert_to_target' from 'qiskit.providers'
```

This indicates a version mismatch. See [VERSION_FIX.md](VERSION_FIX.md) for detailed troubleshooting.

**Quick fix**:
```bash
# Uninstall incompatible versions
pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu

# Install compatible versions
pip install "qiskit>=2.0.0,<3.0.0" "qiskit-aer>=0.17.0,<0.18.0"
```

### Environment Issues

If using conda and having issues:
```bash
# Remove old environment
conda env remove -n HHL

# Create fresh environment
conda env create -f environment.yml
conda activate HHL
```

### GPU Not Available

If GPU is not working:
1. Check CUDA installation: `nvcc --version`
2. Run diagnostic: `python scripts/check_gpu_support.py`
3. See [docs/INSTALL_GPU.md](docs/INSTALL_GPU.md) for build instructions

### Verification

```bash
# Verify installation
python -c "from QLS.hhl import HHL; print('✓ OK')"
```

### Out of Memory

For large matrices:
- Reduce `NUM_WORK_QUBITS` in configuration
- Use `LIGHTWEIGHT_MODE=True`
- Increase system RAM or use GPU

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Qiskit team for the excellent quantum computing framework
- Original HHL algorithm: Harrow, A. W., Hassidim, A., Lloyd, S. (2009)

## 📧 Contact

- **Website**: [Kooshan.info](https://Kooshan.info)

For questions, issues, or contributions, please visit the project repository or contact through the website.

---

**Note**: Logs directory is gitignored. All runtime logs are stored locally and not committed to version control.
