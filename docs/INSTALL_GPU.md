# Installing GPU Support for Qiskit Aer

## Current Situation

- **Your Qiskit version**: 2.2.1
- **Available qiskit-aer-gpu**: 0.15.1 (latest)
- **Problem**: qiskit-aer-gpu 0.15.1 is built for Qiskit 1.x, not 2.x
- **Result**: Installing qiskit-aer-gpu will cause import errors

## Why GPU Doesn't Work

The `qiskit-aer-gpu` package on PyPI is outdated:
- Latest: 0.15.1 (for Qiskit 1.x)
- Needed: 0.17.x+ (for Qiskit 2.x) - **NOT AVAILABLE**

When you try to install `qiskit-aer-gpu` with Qiskit 2.x, you'll get:
```
ImportError: cannot import name 'convert_to_target' from 'qiskit.providers'
```

## Solutions

### Option 1: Build from Source (Recommended for Qiskit 2.x)

This is the **only way** to get GPU support with Qiskit 2.x currently.

#### Prerequisites

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git
# Or on macOS:
# brew install cmake

# CUDA toolkit (if not already installed)
# Check: nvcc --version
# Install from: https://developer.nvidia.com/cuda-downloads
```

#### Build Steps

```bash
# Clone the repository
git clone https://github.com/Qiskit/qiskit-aer.git
cd qiskit-aer

# Install Python build dependencies
pip install -r requirements-dev.txt

# Build with GPU support
python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

# Install the built wheel
pip install dist/qiskit_aer-*.whl --force-reinstall
```

#### Verify Installation

```bash
python examples/check_gpu_support.py
```

You should see:
```
✓ GPU support: AVAILABLE
```

### Option 2: Downgrade to Qiskit 1.x (Not Recommended)

⚠️ **Warning**: This will downgrade your entire Qiskit ecosystem and may break other code.

```bash
# Uninstall current versions
pip uninstall qiskit qiskit-aer -y

# Install compatible versions
pip install qiskit==1.1.1 qiskit-aer-gpu==0.15.1
```

**Pros:**
- Quick installation
- GPU works immediately

**Cons:**
- Lose Qiskit 2.x features
- May break other dependencies
- Security/bug fixes missing

### Option 3: Use CPU (Current Setup)

Your code already works with CPU backend. It's slower but functional.

**Performance:**
- 4×4 matrix: ~0.1 sec
- 8×8 matrix: ~1 sec
- 16×16 matrix: 5-10 minutes
- 32×32 matrix: Hours

**When to use:**
- Development and testing
- Small-medium problems
- When GPU setup is too complex

## Checking GPU Support

After installation, verify GPU support:

```bash
# Quick test
python examples/gpu_test.py

# Detailed diagnostic
python examples/check_gpu_support.py
```

## Troubleshooting

### Build Fails

**Error: "CUDA not found"**
```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Error: "CMake version too old"**
```bash
# Install newer CMake
# Ubuntu/Debian:
sudo apt-get install cmake=3.20+

# Or build from source:
wget https://cmake.org/files/v3.27/cmake-3.27.0.tar.gz
tar -xzf cmake-3.27.0.tar.gz
cd cmake-3.27.0
./bootstrap && make && sudo make install
```

### Import Errors After Installation

If you get import errors:
```bash
# Verify installation
python -c "from qiskit_aer import AerSimulator; print('OK')"

# Check available devices
python -c "from qiskit_aer import AerSimulator; b=AerSimulator(); print(b.available_devices())"
```

## Monitoring for Updates

Check periodically for GPU package updates:

```bash
pip index versions qiskit-aer-gpu
```

When qiskit-aer-gpu 0.17.x+ becomes available, you can simply:
```bash
pip install qiskit-aer-gpu --upgrade
```

## Resources

- **Qiskit Aer GitHub**: https://github.com/Qiskit/qiskit-aer
- **Build Instructions**: https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md
- **Qiskit Slack**: https://qisk.it/join-slack
- **Issues**: https://github.com/Qiskit/qiskit-aer/issues

## Summary

| Solution | Difficulty | GPU Support | Qiskit 2.x Compatible |
|----------|-----------|-------------|----------------------|
| Build from source | Medium | ✓ Yes | ✓ Yes |
| Downgrade to 1.x | Easy | ✓ Yes | ✗ No |
| Use CPU | Easy | ✗ No | ✓ Yes |

**Recommendation**: For Qiskit 2.x, build from source. For quick testing, use CPU.
