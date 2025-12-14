#!/bin/bash
# Fix Qiskit/qiskit-aer version compatibility issues

set -e

echo "=========================================="
echo "Qiskit/qiskit-aer Version Fix Script"
echo "=========================================="
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠ Warning: Not in a conda environment"
    echo "  This script is designed for conda environments"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "Step 1: Uninstalling incompatible versions..."
pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu 2>/dev/null || true

echo ""
echo "Step 2: Installing compatible versions..."
pip install 'qiskit>=2.0.0,<3.0.0' 'qiskit-aer>=0.17.0,<0.18.0'

echo ""
echo "Step 3: Verifying installation..."
python -c "
import qiskit
import qiskit_aer
print(f'✓ Qiskit: {qiskit.__version__}')
print(f'✓ qiskit-aer: {qiskit_aer.__version__}')

# Test import
try:
    from qiskit_aer import AerSimulator
    print('✓ Import test successful')
except Exception as e:
    print(f'✗ Import test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Version fix completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Version fix failed. Please check errors above."
    echo "=========================================="
    exit 1
fi
