#!/bin/bash
# Comprehensive GPU setup and testing script for Qiskit Aer
# Combines version fixing, GPU build, diagnostics, and testing

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Function: Fix Qiskit/qiskit-aer version compatibility
fix_versions() {
    print_section "Qiskit/qiskit-aer Version Fix"
    
    # Check if we're in a conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "Not in a conda environment"
        print_info "This script is designed for conda environments"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    else
        print_success "Conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    print_info "Step 1: Uninstalling incompatible versions..."
    pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu 2>/dev/null || true
    
    print_info "Step 2: Installing compatible versions..."
    pip install 'qiskit>=2.0.0,<3.0.0' 'qiskit-aer>=0.17.0,<0.18.0'
    
    print_info "Step 3: Verifying installation..."
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
        print_success "Version fix completed successfully!"
        return 0
    else
        print_error "Version fix failed. Please check errors above."
        return 1
    fi
}

# Function: Build qiskit-aer with GPU support
build_gpu_aer() {
    print_section "Building qiskit-aer with GPU Support"
    
    # Check if we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "HHL" ]]; then
        print_warning "Not in HHL conda environment"
        print_info "Please run: conda activate HHL"
        print_info "Then run this script again"
        return 1
    fi
    
    # Check prerequisites
    print_info "[1/7] Checking prerequisites..."
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. Install CUDA toolkit."
        return 1
    fi
    print_success "CUDA found: $(nvcc --version | grep release)"
    
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found. Install CMake."
        return 1
    fi
    print_success "CMake found: $(cmake --version | head -1)"
    
    # Check Python environment
    print_success "Python: $(python --version)"
    print_success "Python path: $(which python)"
    
    # Set CUDA paths
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    print_info "[2/7] Installing core build dependencies..."
    pip install -q scikit-build cmake ninja
    
    print_info "[3/7] Setting up build directory..."
    BUILD_DIR="/tmp/qiskit-aer"
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory $BUILD_DIR does not exist"
        print_info "Please clone qiskit-aer repository first:"
        print_info "  git clone https://github.com/Qiskit/qiskit-aer.git $BUILD_DIR"
        return 1
    fi
    cd "$BUILD_DIR"
    
    # Clean previous build
    print_info "Cleaning previous build..."
    rm -rf _skbuild dist build *.egg-info 2>/dev/null || true
    
    print_info "[4/7] Installing Python build dependencies..."
    print_info "Installing pybind11 (required for bindings)..."
    pip install -q pybind11
    print_info "Installing other requirements..."
    if [ -f "requirements-dev.txt" ]; then
        pip install -q -r requirements-dev.txt
    fi
    
    print_info "[5/7] Building qiskit-aer with GPU support..."
    print_info "This will take 10-30 minutes..."
    print_info "Building with CUDA backend..."
    python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA
    
    print_info "[6/7] Installing built package..."
    WHEEL_FILE=$(ls dist/qiskit_aer-*.whl 2>/dev/null | head -1)
    if [ -z "$WHEEL_FILE" ]; then
        print_error "Wheel file not found in dist/"
        print_error "Build may have failed. Check the output above for errors."
        return 1
    fi
    
    print_success "Found wheel: $WHEEL_FILE"
    print_info "Installing (this will replace CPU-only version)..."
    pip install "$WHEEL_FILE" --force-reinstall --no-deps
    
    print_info "Reinstalling dependencies..."
    pip install -q qiskit-aer
    
    print_info "[7/7] Verifying installation..."
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
    
    if [ $? -eq 0 ]; then
        print_success "Build Complete!"
        print_info "Next steps:"
        print_info "  1. Run GPU diagnostics: $0 check"
        print_info "  2. Run GPU test: $0 test"
        return 0
    else
        print_error "Build verification failed"
        return 1
    fi
}

# Function: Check GPU support (diagnostics)
check_gpu_support() {
    print_section "Qiskit Aer GPU Support Diagnostic"
    
    python3 << 'PYTHON_SCRIPT'
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

print("="*70)
print("QISKIT AER GPU SUPPORT DIAGNOSTIC")
print("="*70)

# Check 1: Import qiskit-aer
print("\n1. Checking qiskit-aer installation...")
try:
    import qiskit_aer
    print(f"   ✓ qiskit-aer version: {qiskit_aer.__version__}")
except ImportError:
    print("   ✗ qiskit-aer is not installed")
    print("   Install with: pip install qiskit-aer")
    sys.exit(1)

# Check 2: Check available devices
print("\n2. Checking available devices...")
try:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    available_devices = backend.available_devices()
    print(f"   Available devices: {available_devices}")
    
    if 'GPU' in available_devices:
        print("   ✓ GPU support is available!")
    else:
        print("   ✗ GPU support is NOT available (CPU-only build)")
        print("   This means qiskit-aer was compiled without GPU support.")
except Exception as e:
    print(f"   ✗ Error checking devices: {e}")

# Check 3: Check version compatibility
print("\n3. Checking version compatibility...")
try:
    import qiskit
    qiskit_version = qiskit.__version__
    print(f"   Qiskit version: {qiskit_version}")
    
    # Check if qiskit-aer-gpu is available
    import subprocess
    result = subprocess.run(['pip', 'index', 'versions', 'qiskit-aer-gpu'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        # Parse the output to get latest version
        lines = result.stdout.strip().split('\n')
        if len(lines) > 0 and 'Available versions:' in lines[0]:
            latest_gpu = lines[0].split('(')[1].split(')')[0]
            print(f"   Latest qiskit-aer-gpu: {latest_gpu}")
            
            # Check if versions are compatible
            qiskit_major = int(qiskit_version.split('.')[0])
            if qiskit_major >= 2:
                print("   ⚠ WARNING: Qiskit 2.x detected")
                print("   qiskit-aer-gpu 0.15.1 is built for Qiskit 1.x")
                print("   They are INCOMPATIBLE - installing will cause errors")
    else:
        print("   Could not check qiskit-aer-gpu availability")
except Exception as e:
    print(f"   Could not check version compatibility: {e}")

# Check 4: Try to create GPU backend
print("\n4. Testing GPU backend creation...")
try:
    from qiskit_aer import AerSimulator
    backend_gpu = AerSimulator(method='statevector', device='GPU')
    print("   ✓ GPU backend created successfully!")
    
    # Test with a simple circuit
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    result = backend_gpu.run(qc).result()
    print("   ✓ GPU backend test circuit executed successfully!")
    
except Exception as e:
    print(f"   ✗ GPU backend creation failed: {e}")
    if "not supported" in str(e):
        print("\n   DIAGNOSIS: qiskit-aer is installed but compiled without GPU support.")
        print("\n   SOLUTIONS:")
        print("   Option 1: Install qiskit-aer-gpu (ONLY if using Qiskit 1.x)")
        print("     ⚠ WARNING: qiskit-aer-gpu 0.15.1 is incompatible with Qiskit 2.x")
        print("     Check: pip index versions qiskit-aer-gpu")
        print("     Install: pip install qiskit-aer-gpu  (only for Qiskit 1.x)")
        print("\n   Option 2: Build qiskit-aer from source with GPU support (RECOMMENDED)")
        print("     See: https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md")
        print("     Requires: CUDA toolkit, CMake, C++ compiler")
        print("     This is the only way to get GPU support with Qiskit 2.x")
        print("\n   Option 3: Use CPU backend (current fallback)")
        print("     Your code will automatically use CPU when GPU is unavailable.")
        print("     Works fine for small-medium problems (≤ 8×8 matrices)")

# Check 5: GPU hardware detection
print("\n5. Checking GPU hardware...")
try:
    from QLS.gpu_utils import get_gpu_info, print_gpu_info
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)
except Exception as e:
    print(f"   Could not check GPU hardware: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

try:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    devices = backend.available_devices()
    
    if 'GPU' in devices:
        print("✓ GPU support: AVAILABLE")
        print("  You can use GPU backends for faster simulations!")
    else:
        print("✗ GPU support: NOT AVAILABLE")
        print("  qiskit-aer is CPU-only. Install qiskit-aer-gpu or build from source.")
        
        # Check if GPUs are available
        try:
            from QLS.gpu_utils import get_gpu_info
            gpu_info = get_gpu_info()
            if gpu_info.get('nvidia_smi') or gpu_info.get('cuda_available'):
                print("  Note: GPUs are detected on your system, but qiskit-aer")
                print("        was not compiled with GPU support.")
        except:
            pass
except Exception as e:
    print(f"✗ Could not determine GPU support status: {e}")

print("="*70)
PYTHON_SCRIPT
}

# Function: Test GPU functionality
gpu_test() {
    print_section "GPU Compatibility Test"
    
    python3 << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
import time

# Add src directory to Python path
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from QLS.gpu_utils import create_gpu_backend, print_gpu_info, get_gpu_info, create_cpu_backend
from qiskit import QuantumCircuit

print("="*70)
print("GPU COMPATIBILITY TEST")
print("="*70)

# Get and print GPU information
print("\n1. Checking GPU availability...")
gpu_info = get_gpu_info()
print_gpu_info(gpu_info)

# Test GPU backend creation
print("\n2. Testing GPU backend creation...")
try:
    backend_gpu, use_gpu = create_gpu_backend(verbose=True)
    
    if use_gpu:
        print("\n✓ GPU backend is working!")
        
        # Run a simple test circuit
        print("\n3. Running test circuit on GPU...")
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        start = time.time()
        result = backend_gpu.run(qc, shots=1000).result()
        elapsed = time.time() - start
        
        print(f"✓ Test circuit executed successfully in {elapsed:.3f} seconds")
        print(f"  Counts: {result.get_counts()}")
    else:
        print("\n⚠ GPU not available, using CPU backend")
        
except Exception as e:
    print(f"\n✗ Error creating GPU backend: {e}")
    print("Falling back to CPU...")
    backend_cpu = create_cpu_backend(verbose=True)

# Test CPU backend for comparison
print("\n4. Testing CPU backend...")
backend_cpu = create_cpu_backend(verbose=True)
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

start = time.time()
result = backend_cpu.run(qc, shots=1000).result()
elapsed = time.time() - start

print(f"✓ CPU test circuit executed in {elapsed:.3f} seconds")
print(f"  Counts: {result.get_counts()}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
PYTHON_SCRIPT
}

# Function: Show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  fix          Fix Qiskit/qiskit-aer version compatibility issues"
    echo "  build        Build qiskit-aer from source with GPU support"
    echo "  check        Run GPU support diagnostics"
    echo "  test         Run GPU compatibility test"
    echo "  all          Run all steps in sequence (fix -> build -> check -> test)"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 fix       # Fix version compatibility"
    echo "  $0 build     # Build GPU-enabled qiskit-aer"
    echo "  $0 check     # Check GPU support status"
    echo "  $0 test      # Test GPU functionality"
    echo "  $0 all       # Run complete setup"
}

# Main script logic
main() {
    cd "$PROJECT_ROOT"
    
    case "${1:-help}" in
        fix)
            fix_versions
            ;;
        build)
            build_gpu_aer
            ;;
        check)
            check_gpu_support
            ;;
        test)
            gpu_test
            ;;
        all)
            print_section "Complete GPU Setup"
            print_info "Running all setup steps..."
            fix_versions && \
            build_gpu_aer && \
            check_gpu_support && \
            gpu_test
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
