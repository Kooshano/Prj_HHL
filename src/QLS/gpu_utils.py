"""
GPU utility functions for Qiskit Aer backend configuration.

This module provides utilities for setting up and managing GPU backends
for quantum simulations, with automatic fallback to CPU if GPU is unavailable.
"""

import subprocess
from typing import Optional, Tuple

# Try to import qiskit-aer, but handle version compatibility issues gracefully
QISKIT_AER_AVAILABLE = False
_import_error = None

try:
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit
    QISKIT_AER_AVAILABLE = True
except (ImportError, AttributeError) as e:
    QISKIT_AER_AVAILABLE = False
    _import_error = str(e)
    # Create dummy classes for type hints when import fails
    class AerSimulator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"qiskit-aer is not available due to version compatibility issues.\n"
                f"Error: {_import_error}\n\n"
                f"To fix: pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu && "
                f"pip install 'qiskit>=2.0.0,<3.0.0' 'qiskit-aer>=0.17.0,<0.18.0'"
            )
    class QuantumCircuit:
        pass


def get_gpu_info() -> dict:
    """
    Get information about available GPUs.
    
    Returns:
        dict: Dictionary containing GPU information from nvidia-smi and PyTorch (if available)
    """
    gpu_info = {
        'nvidia_smi': None,
        'pytorch_cuda': None,
        'cuda_available': False
    }
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,index,memory.total,memory.free,memory.used', 
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'name': parts[0],
                            'index': parts[1],
                            'memory_total': parts[2],
                            'memory_free': parts[3],
                            'memory_used': parts[4]
                        })
            gpu_info['nvidia_smi'] = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        pass
    
    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['pytorch_cuda'] = {
                'device_count': torch.cuda.device_count(),
                'devices': []
            }
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info['pytorch_cuda']['devices'].append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': props.total_memory / 1e9,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
    except ImportError:
        pass
    except Exception:
        pass
    
    return gpu_info


def print_gpu_info(gpu_info: Optional[dict] = None):
    """
    Print GPU information in a formatted way.
    
    Args:
        gpu_info: Optional GPU info dict. If None, will fetch it.
    """
    if gpu_info is None:
        gpu_info = get_gpu_info()
    
    print("="*60)
    print("GPU Information")
    print("="*60)
    
    if gpu_info.get('nvidia_smi'):
        print("\nNVIDIA GPU Details (nvidia-smi):")
        for gpu in gpu_info['nvidia_smi']:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Total Memory: {gpu['memory_total']}")
            print(f"    Free Memory:  {gpu['memory_free']}")
            print(f"    Used Memory:  {gpu['memory_used']}")
    else:
        print("\nNVIDIA GPU Details: Not available (nvidia-smi not found or no GPUs)")
    
    if gpu_info.get('pytorch_cuda'):
        print(f"\nPyTorch CUDA Information:")
        print(f"  CUDA Available: True")
        print(f"  CUDA Device Count: {gpu_info['pytorch_cuda']['device_count']}")
        for device in gpu_info['pytorch_cuda']['devices']:
            print(f"  GPU {device['index']}: {device['name']}")
            print(f"    Memory: {device['memory_total_gb']:.2f} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
    elif gpu_info.get('cuda_available') is False:
        print("\nPyTorch CUDA: Not available")
    
    print("="*60)


def create_backend(
    device: str = 'auto',
    method: str = 'statevector',
    precision: str = 'double',
    verbose: bool = True
) -> Tuple[AerSimulator, bool]:
    """
    Create a Qiskit Aer backend with automatic GPU detection and fallback.
    
    Args:
        device: Device to use ('auto', 'GPU', or 'CPU'). 'auto' will try GPU first.
        method: Simulation method ('statevector', 'density_matrix', etc.)
        precision: Numerical precision ('single' or 'double')
        verbose: Whether to print GPU information and status
    
    Returns:
        Tuple[AerSimulator, bool]: (backend, use_gpu) where use_gpu indicates if GPU is actually being used
    
    Raises:
        ImportError: If qiskit-aer is not available or has version compatibility issues
    """
    if not QISKIT_AER_AVAILABLE:
        error_msg = (
            f"qiskit-aer is not available or has version compatibility issues.\n"
            f"Error: {_import_error}\n\n"
            f"To fix this:\n"
            f"1. Run: pip uninstall -y qiskit qiskit-aer qiskit-aer-gpu\n"
            f"2. Run: pip install 'qiskit>=2.0.0,<3.0.0' 'qiskit-aer>=0.17.0,<0.18.0'\n"
            f"3. Or use the fix script: ./fix_versions.sh"
        )
        raise ImportError(error_msg)
    
    use_gpu = False
    backend = None
    
    # Determine device preference
    if device == 'auto':
        # Check if GPU is available
        gpu_info = get_gpu_info()
        has_gpu = (gpu_info.get('nvidia_smi') is not None or 
                  gpu_info.get('cuda_available', False))
        device_pref = 'GPU' if has_gpu else 'CPU'
    else:
        device_pref = device.upper()
    
    # Try to create GPU backend if requested
    if device_pref == 'GPU':
        try:
            backend_gpu = AerSimulator(
                method=method,
                device='GPU',
                precision=precision
            )
            
            # Test if GPU actually works by running a simple circuit
            test_qc = QuantumCircuit(2)
            test_qc.h(0)
            test_qc.cx(0, 1)
            test_result = backend_gpu.run(test_qc).result()
            
            backend = backend_gpu
            use_gpu = True
            
            if verbose:
                print("✓ GPU backend successfully initialized and tested!")
                
        except Exception as e:
            if verbose:
                error_msg = str(e)
                if "GPU" in error_msg and "not supported" in error_msg:
                    print(f"⚠ GPU backend not available: {e}")
                    print("   This usually means qiskit-aer was compiled without GPU support.")
                    print("   To enable GPU support, you need to:")
                    print("   1. Install qiskit-aer-gpu (if available for your Qiskit version), OR")
                    print("   2. Build qiskit-aer from source with GPU support enabled")
                    print("   See docs/GPU_STATUS.md for more details.")
                else:
                    print(f"⚠ GPU backend not available or not working: {e}")
                print("Falling back to CPU backend...")
            
            # Fallback to CPU
            try:
                backend = AerSimulator(
                    method=method,
                    device='CPU',
                    precision=precision
                )
                use_gpu = False
            except Exception as e2:
                raise RuntimeError(f"Failed to create both GPU and CPU backends. GPU error: {e}, CPU error: {e2}")
    
    else:  # CPU
        backend = AerSimulator(
            method=method,
            device='CPU',
            precision=precision
        )
        use_gpu = False
    
    if verbose:
        print(f"\nBackend Configuration:")
        print(f"  Backend name: {backend.name}")
        print(f"  Available devices: {backend.available_devices()}")
        print(f"  Actually using GPU: {use_gpu}")
        print(f"  Method: {method}")
        print(f"  Precision: {precision}")
    
    return backend, use_gpu


def create_gpu_backend(
    method: str = 'statevector',
    precision: str = 'double',
    verbose: bool = True
) -> Tuple[AerSimulator, bool]:
    """
    Convenience function to create a GPU backend (with CPU fallback).
    
    Args:
        method: Simulation method ('statevector', 'density_matrix', etc.)
        precision: Numerical precision ('single' or 'double')
        verbose: Whether to print GPU information and status
    
    Returns:
        Tuple[AerSimulator, bool]: (backend, use_gpu) where use_gpu indicates if GPU is actually being used
    """
    return create_backend(device='auto', method=method, precision=precision, verbose=verbose)


def create_cpu_backend(
    method: str = 'statevector',
    precision: str = 'double',
    verbose: bool = False
) -> AerSimulator:
    """
    Convenience function to create a CPU backend.
    
    Args:
        method: Simulation method ('statevector', 'density_matrix', etc.)
        precision: Numerical precision ('single' or 'double')
        verbose: Whether to print backend information
    
    Returns:
        AerSimulator: CPU backend
    """
    backend = AerSimulator(
        method=method,
        device='CPU',
        precision=precision
    )
    if verbose:
        print(f"CPU backend created: {backend.name}")
    return backend
