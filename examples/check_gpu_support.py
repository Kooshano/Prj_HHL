#!/usr/bin/env python
"""
Diagnostic script to check GPU support status for Qiskit Aer.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from QLS.gpu_utils import get_gpu_info, print_gpu_info
gpu_info = get_gpu_info()
print_gpu_info(gpu_info)

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
        if gpu_info.get('nvidia_smi') or gpu_info.get('cuda_available'):
            print("  Note: GPUs are detected on your system, but qiskit-aer")
            print("        was not compiled with GPU support.")
except Exception as e:
    print(f"✗ Could not determine GPU support status: {e}")

print("="*70)
