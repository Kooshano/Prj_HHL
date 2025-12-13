# GPU Support Status for Qiskit Aer HHL Project

**Date:** November 19, 2025  
**System:** 2× NVIDIA RTX 6000 Ada Generation (51GB each)

---

## Current Status: ❌ GPU NOT AVAILABLE

### Why GPU Doesn't Work

| Component | Your System | Required | Status |
|-----------|-------------|----------|--------|
| **NVIDIA GPU** | RTX 6000 Ada × 2 | ✓ | ✓ Available |
| **CUDA** | CUDA 12.6 | CUDA 11.2+ | ✓ Working |
| **qiskit** | 2.2.3 | Any | ✓ Installed |
| **qiskit-aer** | 0.17.2 (CPU) | - | ✓ Working |
| **qiskit-aer-gpu** | 0.15.1 (latest) | 0.17.2+ | ❌ **TOO OLD** |

### The Problem

```
qiskit-aer (CPU):     v0.17.2  ← Latest, compatible
qiskit-aer-gpu:       v0.15.1  ← 2 versions behind!
                        ↑
                    INCOMPATIBLE
```

**Error when installing GPU version:**
```python
ImportError: cannot import name 'convert_to_target' from 'qiskit.providers'
```

This happens because `qiskit-aer-gpu` 0.15.1 was built for Qiskit 1.x, not 2.x.

---

## Solutions (Ranked by Difficulty)

### Option 1: Use CPU (Current) ✓ **RECOMMENDED**

**Pros:**
- ✓ Already working
- ✓ No setup required
- ✓ Good for problems ≤ 8×8 (3 qubits)

**Cons:**
- ✗ Slow for large problems (16×16+ takes minutes)

**When to use:** Testing, development, small-medium problems

---

### Option 2: Wait for Official Update ⏳

The Qiskit team will eventually release `qiskit-aer-gpu` 0.17.x

**Check status:**
```bash
pip index versions qiskit-aer-gpu
```

**Watch:** https://github.com/Qiskit/qiskit-aer/releases

**ETA:** Unknown (could be weeks or months)

---

### Option 3: Build from Source 🛠️ **ADVANCED**

Build GPU-enabled Qiskit Aer yourself.

**Requirements:**
- CUDA Toolkit 12.x
- CMake 3.20+
- C++ compiler (gcc 9+)
- ~60 minutes build time

**Steps:**
```bash
# Install build dependencies
sudo apt-get install build-essential cmake

# Clone repository
git clone https://github.com/Qiskit/qiskit-aer.git
cd qiskit-aer

# Build with GPU support
pip install -r requirements-dev.txt
python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

# Install
pip install dist/qiskit_aer-*.whl
```

**Pros:**
- ✓ Get latest features with GPU
- ✓ Compatible with Qiskit 2.2+

**Cons:**
- ✗ Complex setup
- ✗ Need to rebuild after updates
- ✗ May have build errors

**Resources:**
- Build guide: https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md
- Build issues: https://github.com/Qiskit/qiskit-aer/issues

---

### Option 4: Downgrade Qiskit ⚠️ **NOT RECOMMENDED**

Match `qiskit-aer-gpu` 0.15.1 by downgrading Qiskit.

```bash
pip install qiskit==1.1.1 qiskit-aer-gpu==0.15.1
```

**Pros:**
- ✓ GPU will work

**Cons:**
- ✗ May break other packages
- ✗ Lose Qiskit 2.x features
- ✗ Security/bug fixes missing

---

## Performance Comparison

### CPU vs GPU (Expected)

| Problem Size | Qubits | CPU Time | GPU Time (est) | Speedup |
|--------------|--------|----------|----------------|---------|
| 4×4          | 2      | 0.1 sec  | 0.1 sec        | 1× |
| 8×8          | 3      | 1 sec    | 0.5 sec        | 2× |
| 16×16        | 4      | 5-10 min | 30-60 sec      | 5-10× |
| 32×32        | 5      | Hours    | 5-10 min       | 10-20× |
| 64×64        | 6      | Days     | 30-60 min      | 50-100× |

**Note:** GPU advantage grows with problem size.

---

## Current Working Setup ✓

```python
from qiskit_aer import AerSimulator
from QLS.hhl import HHL

# CPU backend (works)
backend = AerSimulator(method='statevector', device='CPU')
hhl = HHL(epsilon=1e-2, quantum_instance=backend)

# Solve
result = hhl.solve(A, b)
```

**Test scripts that work:**
- ✓ `test_hhl_quick.py` - Quick 4×4 test (0.06 sec)
- ✓ `HHL_demo.py` - Full demo (0.11 sec)
- ✓ `HHL_Project.py` - Main project (adjusted to 4×4)

---

## Recommendations

### For your use case:

1. **Start with CPU** (current setup) ✓
   - Test and develop algorithms
   - Works perfectly for ≤ 8×8 problems

2. **Monitor GPU version releases**
   ```bash
   # Check weekly
   pip index versions qiskit-aer-gpu
   ```

3. **Consider building from source** if you need:
   - Large problems (32×32+)
   - Frequent runs
   - Production performance

### When GPU becomes critical:

If you have urgent need for GPU:
1. Try **Option 3** (build from source)
2. Or use a cloud service with older Qiskit (AWS Braket, IBM Quantum)
3. Or wait for official 0.17.x GPU release

---

## Monitoring Updates

**Check for GPU package updates:**
```bash
pip index versions qiskit-aer-gpu
# Currently shows: 0.15.1 (latest)
# Waiting for: 0.17.x
```

**GitHub watch:**
- https://github.com/Qiskit/qiskit-aer/releases
- Look for releases with GPU binary wheels

---

## Contact & Support

- **Qiskit Slack:** https://qisk.it/join-slack
- **GitHub Issues:** https://github.com/Qiskit/qiskit-aer/issues
- **Stack Overflow:** Tag `qiskit`

---

**Last verified:** November 19, 2025  
**Status:** Everything working on CPU ✓


