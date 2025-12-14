#!/usr/bin/env python
"""
Quick test script to verify GPU support is working correctly.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from QLS.gpu_utils import create_gpu_backend, print_gpu_info, get_gpu_info, create_cpu_backend
from qiskit import QuantumCircuit
import time

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
