#!/bin/bash
# Fixed script to build qiskit-aer with GPU support
# Make sure you're in the HHL conda environment before running

set -e  # Exit on error

echo "=========================================="
echo "Building qiskit-aer with GPU Support"
echo "=========================================="
echo ""

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "HHL" ]]; then
    echo "⚠ WARNING: Not in HHL conda environment"
    echo "Please run: conda activate HHL"
    echo "Then run this script again"
    exit 1
fi

# Check prerequisites
echo "[1/7] Checking prerequisites..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit."
    exit 1
fi
echo "  ✓ CUDA found: $(nvcc --version | grep release)"

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Install CMake."
    exit 1
fi
echo "  ✓ CMake found: $(cmake --version | head -1)"

# Check Python environment
echo "  ✓ Python: $(python --version)"
echo "  ✓ Python path: $(which python)"

# Set CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo "[2/7] Installing core build dependencies..."
pip install -q scikit-build cmake ninja

echo ""
echo "[3/7] Setting up build directory..."
BUILD_DIR="/tmp/qiskit-aer"
cd "$BUILD_DIR"

# Clean previous build
echo "  → Cleaning previous build..."
rm -rf _skbuild dist build *.egg-info 2>/dev/null || true

echo ""
echo "[4/7] Installing Python build dependencies..."
echo "  → Installing pybind11 (required for bindings)..."
pip install -q pybind11
echo "  → Installing other requirements..."
pip install -q -r requirements-dev.txt

echo ""
echo "[5/7] Building qiskit-aer with GPU support..."
echo "  → This will take 10-30 minutes..."
echo "  → Building with CUDA backend..."
echo "  → Progress will be shown below..."
python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

echo ""
echo "[6/7] Installing built package..."
WHEEL_FILE=$(ls dist/qiskit_aer-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel file not found in dist/"
    echo "Build may have failed. Check the output above for errors."
    exit 1
fi

echo "  → Found wheel: $WHEEL_FILE"
echo "  → Installing (this will replace CPU-only version)..."
pip install "$WHEEL_FILE" --force-reinstall --no-deps

echo "  → Reinstalling dependencies..."
pip install -q qiskit-aer

echo ""
echo "[7/7] Verifying installation..."
python -c "
from qiskit_aer import AerSimulator
backend = AerSimulator()
devices = backend.available_devices()
print(f'Available devices: {devices}')
if 'GPU' in devices:
    print('✓ GPU support is now available!')
    print('✓ Build successful!')
else:
    print('✗ GPU support not found. Build may have failed.')
    print('  Available devices:', devices)
    exit(1)
"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python examples/check_gpu_support.py"
echo "  2. Test your code: python HHL_Project.py"
echo "  3. You should see 'Backend created (GPU)' in the output"
