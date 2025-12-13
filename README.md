# HHL Quantum Linear Solver Project

This project implements the Harrow-Hassidim-Lloyd (HHL) quantum algorithm for solving linear systems of equations.

## Project Structure

```
Prj_HHL/
├── QLS/                    # Main quantum linear solver package
│   ├── hhl.py              # HHL quantum solver implementation
│   ├── linear_solver.py    # Base linear solver interface
│   ├── numpy_linear_solver.py  # Classical NumPy-based solver
│   ├── matrices/           # Matrix implementations
│   │   ├── linear_system_matrix.py
│   │   ├── numpy_matrix.py
│   │   └── tridiagonal_toeplitz.py
│   └── observables/        # Quantum observables
│       ├── absolute_average.py
│       ├── linear_system_observable.py
│       └── matrix_functional.py
├── notebooks/              # Jupyter notebooks
│   ├── HHL Project.ipynb
│   ├── hhl_test.ipynb
│   ├── hhl_tutorial-3.ipynb
│   └── Test.ipynb
├── examples/               # Example scripts and demos
│   └── HHL_demo.py
├── tests/                  # Test files
│   ├── test_hhl_quick.py
│   ├── test.py
│   ├── test_tridiagonal.py
│   └── VERIFY_ALL.py
├── data/                   # Data files (CSV datasets)
│   ├── Fish.csv
│   └── housing.csv
├── logs/                   # Runtime logs
│   └── hhl_runs_log.json
├── docs/                   # Documentation
│   └── GPU_STATUS.md
├── HHL_Project.py          # Main project script
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── .gitignore             # Git ignore rules

```

## Setup

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate HHL
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

### Main Project Script
```bash
python HHL_Project.py
```

### Quick Demo
```bash
python examples/HHL_demo.py
```

### Quick Test
```bash
python tests/test_hhl_quick.py
```

## Key Features

- Quantum HHL algorithm implementation using Qiskit
- Classical NumPy-based linear solver for comparison
- Support for Hermitian and non-Hermitian matrices
- GPU acceleration support (when available)
- Linear regression using HHL
- Comprehensive testing suite

## Dependencies

- Qiskit (quantum computing framework)
- NumPy (numerical computing)
- SciPy (scientific computing)
- Qiskit Aer (quantum simulator)

See `requirements.txt` for complete dependency list.
